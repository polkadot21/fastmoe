import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F  # For Flash Attention
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 128
MICRO_BATCHES = 2
HIDDEN_DIM = 4096
NUM_HEADS = 32
NUM_EXPERTS_PER_GPU = 2
TOP_K = 2
WARMUP_STEPS = 5
ACTIVE_STEPS = 3

# --- Dead Weight Config ---
# We bloat communication by 20x to match the Expert Compute
COMM_SCALING_FACTOR = 20


# ==========================================
# Model Definition
# ==========================================
class RealSelfAttention(nn.Module):
    """
    Realistic Multi-Head Attention using PyTorch SDPA (Flash Attention).
    Used for Pre-Ops (Layer L) and Post-Ops (Layer L+1).
    """

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "Hidden dim must be divisible by num heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Input x: [Batch, SeqLen, Dim]
        B, S, D = x.shape
        residual = x
        x = self.norm(x)

        # Projections
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention (Hardware aware: uses FlashAttn or MemEfficient automatically)
        # dropout_p=0.0 for profiling stability
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        return out + residual


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LancetBlockFull(nn.Module):
    def __init__(self, hidden_dim, num_experts, world_size, group):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_local_experts = num_experts // world_size
        self.world_size = world_size
        self.group = group

        # 1. Pre-Ops: Real Self-Attention (Layer L) + Gate
        self.attn = RealSelfAttention(hidden_dim, NUM_HEADS)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # 2. Experts: Real MLP Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # 3. Post-Ops: Real Self-Attention (Layer L+1)
        # In Lancet "Forward Extension", we overlap the NEXT layer's attention
        self.next_layer_attn = RealSelfAttention(hidden_dim, NUM_HEADS)

        # Streams
        self.stream_compute = torch.cuda.Stream()
        self.stream_comm = torch.cuda.Stream()
        self.stream_expert = torch.cuda.Stream()

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        B, S, D = x.shape

        # We must chunk along Batch dimension to preserve Sequence for Attention
        chunks = x.chunk(MICRO_BATCHES, dim=0)

        pipeline_buffers = [{} for _ in range(MICRO_BATCHES)]
        outputs = [None] * MICRO_BATCHES

        ev_pre_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_dispatch_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_expert_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_combine_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]

        total_ticks = MICRO_BATCHES + 4

        for tick in range(total_ticks):
            # --- Stage 5: Post-Ops (Compute Stream) ---
            # Real Operation: Next Layer Self-Attention
            mb_post = tick - 4
            if 0 <= mb_post < MICRO_BATCHES:
                buf = pipeline_buffers[mb_post]
                self.stream_compute.wait_event(ev_combine_ready[mb_post])

                with torch.cuda.stream(self.stream_compute):
                    torch.cuda.nvtx.range_push(f"PostOps_MB{mb_post}")
                    with record_function(f"PostOps_MB{mb_post}"):
                        # Reshape flattened tokens back to (MicroBatch, Seq, Dim) for Attn
                        flat_out = buf["combined_output"]
                        mb_size = chunks[mb_post].size(0)  # Get original microbatch size
                        reshaped_in = flat_out.view(mb_size, S, D)

                        out = self.next_layer_attn(reshaped_in)
                        outputs[mb_post] = out
                    torch.cuda.nvtx.range_pop()

            # --- Stage 4: Combine A2A (Comm Stream) ---
            mb_comb = tick - 3
            if 0 <= mb_comb < MICRO_BATCHES:
                buf = pipeline_buffers[mb_comb]
                self.stream_comm.wait_event(ev_expert_ready[mb_comb])

                with torch.cuda.stream(self.stream_comm):
                    torch.cuda.nvtx.range_push(f"A2A_Combine_MB{mb_comb}")
                    with record_function(f"A2A_Combine_MB{mb_comb}"):
                        real_input = buf["expert_output"]

                        # --- Dead Weight Injection ---
                        bloated_input = real_input.repeat(1, COMM_SCALING_FACTOR)
                        bloated_output = torch.empty_like(bloated_input)

                        dist.all_to_all_single(
                            bloated_output,
                            bloated_input,
                            group=self.group,
                            async_op=False,
                        )

                        # Reconstruct valid pipeline data
                        buf["combined_output"] = torch.empty_like(real_input)

                    torch.cuda.nvtx.range_pop()
                ev_combine_ready[mb_comb].record(self.stream_comm)

            # --- Stage 3: Expert Compute (Expert Stream) ---
            mb_exp = tick - 2
            if 0 <= mb_exp < MICRO_BATCHES:
                buf = pipeline_buffers[mb_exp]
                self.stream_expert.wait_event(ev_dispatch_ready[mb_exp])

                with torch.cuda.stream(self.stream_expert):
                    torch.cuda.nvtx.range_push(f"Expert_MB{mb_exp}")
                    with record_function(f"Expert_MB{mb_exp}"):
                        inp = buf["dispatch_output"]
                        # Run Experts on flattened tokens
                        local_splits = inp.chunk(self.num_local_experts)
                        res = [
                            self.experts[i](local_splits[i]) for i in range(self.num_local_experts)
                        ]
                        buf["expert_output"] = torch.cat(res)
                    torch.cuda.nvtx.range_pop()
                ev_expert_ready[mb_exp].record(self.stream_expert)

            # --- Stage 2: Dispatch A2A (Comm Stream) ---
            mb_disp = tick - 1
            if 0 <= mb_disp < MICRO_BATCHES:
                buf = pipeline_buffers[mb_disp]
                self.stream_comm.wait_event(ev_pre_ready[mb_disp])

                with torch.cuda.stream(self.stream_comm):
                    torch.cuda.nvtx.range_push(f"A2A_Dispatch_MB{mb_disp}")
                    with record_function(f"A2A_Dispatch_MB{mb_disp}"):
                        real_input = buf["gated_input"]
                        pipeline_buffers[mb_disp]["dispatch_input"] = real_input

                        # --- Dead Weight Injection ---
                        bloated_input = real_input.repeat(1, COMM_SCALING_FACTOR)
                        bloated_output = torch.empty_like(bloated_input)

                        dist.all_to_all_single(
                            bloated_output,
                            bloated_input,
                            group=self.group,
                            async_op=False,
                        )

                        pipeline_buffers[mb_disp]["dispatch_output"] = torch.empty_like(real_input)

                    torch.cuda.nvtx.range_pop()
                ev_dispatch_ready[mb_disp].record(self.stream_comm)

            # --- Stage 1: Pre-Ops (Compute Stream) ---
            # Real Operation: Layer L Self-Attention
            mb_pre = tick
            if 0 <= mb_pre < MICRO_BATCHES:
                with torch.cuda.stream(self.stream_compute):
                    torch.cuda.nvtx.range_push(f"PreOps_MB{mb_pre}")
                    with record_function(f"PreOps_MB{mb_pre}"):
                        x_chunk = chunks[mb_pre]  # Shape: [MB_Size, Seq, Dim]

                        # Real Attention (Flash)
                        x_attn = self.attn(x_chunk)

                        # Flatten for Gating/MoE: [MB_Size * Seq, Dim]
                        # (Gating typically works on individual tokens)
                        x_flat = x_attn.view(-1, D)

                        logits = self.gate(x_flat)
                        _, indices = torch.topk(logits, k=TOP_K, dim=1)

                        gated = x_flat.clone()
                        pipeline_buffers[mb_pre]["gated_input"] = gated
                    torch.cuda.nvtx.range_pop()
                ev_pre_ready[mb_pre].record(self.stream_compute)

        torch.cuda.current_stream().wait_stream(self.stream_compute)
        final_out = torch.cat(outputs, dim=0)
        return final_out


# ==========================================
# Worker Function
# ==========================================
def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12365"  # Unique port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = LancetBlockFull(
        HIDDEN_DIM, NUM_EXPERTS_PER_GPU * world_size, world_size, dist.group.WORLD
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Data: [Batch, SeqLen, Dim]
    data = torch.randn(BATCH_SIZE, 128, HIDDEN_DIM).cuda()

    def trace_handler(p):
        dist.barrier()
        if rank == 0:
            p.add_metadata(
                "Config",
                f"MB={MICRO_BATCHES}, GPUs={world_size}, Scale={COMM_SCALING_FACTOR}, FlashAttn=True",  # noqa
            )
            abs_path = os.path.abspath("lancet_real_ops_overlap.json")
            p.export_chrome_trace(abs_path)
            print(f"\n[Rank 0] Trace saved to: {abs_path}")

    # Warmup
    if rank == 0:
        print("[Rank 0] Warming up...")
    for _ in range(WARMUP_STEPS):
        loss = model(data).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if rank == 0:
        print(f"[Rank 0] Profiling with Real Flash Attention & {COMM_SCALING_FACTOR}x Comm...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=ACTIVE_STEPS, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
    ) as p:
        total_steps = 1 + 1 + ACTIVE_STEPS
        for _ in range(total_steps):
            loss = model(data).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p.step()

    time.sleep(2)
    dist.barrier()
    dist.destroy_process_group()


# ==========================================
# Execution
# ==========================================
def run_experiment():
    world_size = 2
    print(f"Starting processes for {world_size} GPUs...")
    mp.start_processes(
        worker, args=(world_size,), nprocs=world_size, join=True, start_method="fork"
    )


run_experiment()
