import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 128
MICRO_BATCHES = 4
HIDDEN_DIM = 4096
NUM_HEADS = 32
NUM_EXPERTS_PER_GPU = 2
TOP_K = 2
WARMUP_STEPS = 5
ACTIVE_STEPS = 3
COMM_SCALING_FACTOR = 20


# ==========================================
# Modules
# ==========================================
class RealSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, S, D = x.shape
        residual = x
        x = self.norm(x)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        return out + residual  # Residual 1 is handled here


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# Configurable Lancet Block
# ==========================================
class LancetBlockConfigurable(nn.Module):
    def __init__(
        self, hidden_dim, num_experts, world_size, group, block_name, pre_op_module, post_op_module
    ):
        super().__init__()
        self.block_name = block_name
        self.hidden_dim = hidden_dim
        self.num_local_experts = num_experts // world_size
        self.group = group
        self.world_size = world_size

        # Pre-MoE Norm (Standard Transformer Pre-Norm for the FFN/MoE layer)
        self.moe_norm = nn.LayerNorm(hidden_dim)

        # Pre-Op: Can be Attention, or Identity (if passed None)
        self.pre_ops = pre_op_module if pre_op_module else nn.Identity()

        # Gating
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # Post-Ops (Next Layer)
        self.post_ops = post_op_module if post_op_module else nn.Identity()

        # Streams
        self.stream_compute = torch.cuda.Stream()
        self.stream_comm = torch.cuda.Stream()
        self.stream_expert = torch.cuda.Stream()

    def forward(self, x):
        B, S, D = x.shape
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
            mb_post = tick - 4
            if 0 <= mb_post < MICRO_BATCHES:
                buf = pipeline_buffers[mb_post]
                self.stream_compute.wait_event(ev_combine_ready[mb_post])
                with torch.cuda.stream(self.stream_compute):
                    op_name = self.post_ops.__class__.__name__
                    label = f"{self.block_name}_Post_{op_name}_MB{mb_post}"

                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        # 1. Retrieve MoE Result
                        moe_out = buf["combined_output"]

                        # 2. Retrieve Residual (Input to MoE)
                        residual = buf["gated_input"]

                        # 3. Apply Residual Connection: x + MoE(Norm(x))
                        # Note: We applied Norm in Stage 1 before sending.
                        # So combined_output is Experts(Norm(x)).
                        # We add it to un-normed residual 'gated_input'.
                        post_moe_out = residual + moe_out

                        # 4. Reshape for Next Layer
                        mb_size = chunks[mb_post].size(0)
                        reshaped_in = post_moe_out.view(mb_size, S, D)

                        # 5. Run Next Layer (Post Ops)
                        outputs[mb_post] = self.post_ops(reshaped_in)
                    torch.cuda.nvtx.range_pop()

            # --- Stage 4: Combine A2A (Comm Stream) ---
            mb_comb = tick - 3
            if 0 <= mb_comb < MICRO_BATCHES:
                buf = pipeline_buffers[mb_comb]
                self.stream_comm.wait_event(ev_expert_ready[mb_comb])
                with torch.cuda.stream(self.stream_comm):
                    label = f"{self.block_name}_Combine_MB{mb_comb}"
                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        real_in = buf["expert_output"]
                        bloated_in = real_in.repeat(1, COMM_SCALING_FACTOR)
                        bloated_out = torch.empty_like(bloated_in)
                        dist.all_to_all_single(
                            bloated_out, bloated_in, group=self.group, async_op=False
                        )
                        buf["combined_output"] = torch.empty_like(real_in)
                    torch.cuda.nvtx.range_pop()
                ev_combine_ready[mb_comb].record(self.stream_comm)

            # --- Stage 3: Experts (Expert Stream) ---
            mb_exp = tick - 2
            if 0 <= mb_exp < MICRO_BATCHES:
                buf = pipeline_buffers[mb_exp]
                self.stream_expert.wait_event(ev_dispatch_ready[mb_exp])
                with torch.cuda.stream(self.stream_expert):
                    label = f"{self.block_name}_Experts_MB{mb_exp}"
                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        inp = buf["dispatch_output"]
                        splits = inp.chunk(self.num_local_experts)
                        res = [self.experts[i](splits[i]) for i in range(self.num_local_experts)]
                        buf["expert_output"] = torch.cat(res)
                    torch.cuda.nvtx.range_pop()
                ev_expert_ready[mb_exp].record(self.stream_expert)

            # --- Stage 2: Dispatch A2A (Comm Stream) ---
            mb_disp = tick - 1
            if 0 <= mb_disp < MICRO_BATCHES:
                buf = pipeline_buffers[mb_disp]
                self.stream_comm.wait_event(ev_pre_ready[mb_disp])
                with torch.cuda.stream(self.stream_comm):
                    label = f"{self.block_name}_Dispatch_MB{mb_disp}"
                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        real_in = buf["normed_input_flat"]  # Send the NORMED input
                        bloated_in = real_in.repeat(1, COMM_SCALING_FACTOR)
                        bloated_out = torch.empty_like(bloated_in)
                        dist.all_to_all_single(
                            bloated_out, bloated_in, group=self.group, async_op=False
                        )
                        pipeline_buffers[mb_disp]["dispatch_output"] = torch.empty_like(real_in)
                    torch.cuda.nvtx.range_pop()
                ev_dispatch_ready[mb_disp].record(self.stream_comm)

            # --- Stage 1: Pre-Ops + Gate (Compute Stream) ---
            mb_pre = tick
            if 0 <= mb_pre < MICRO_BATCHES:
                with torch.cuda.stream(self.stream_compute):
                    is_identity = isinstance(self.pre_ops, nn.Identity)
                    op_desc = "GateOnly" if is_identity else "Attn_Gate"
                    label = f"{self.block_name}_Pre_{op_desc}_MB{mb_pre}"

                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        x_chunk = chunks[mb_pre]

                        # 1. Run Pre-Op (e.g., Attention L)
                        # This returns x + Attn(Norm(x)) if it is RealSelfAttention
                        x_proc = self.pre_ops(x_chunk)

                        # 2. Save Residual (x_proc is the input to the MoE block)
                        x_flat = x_proc.view(-1, D)
                        pipeline_buffers[mb_pre]["gated_input"] = (
                            x_flat.clone()
                        )  # Keep un-normed for residual add later

                        # 3. Apply Pre-MoE Norm
                        # Standard Pre-Norm: MoE(Norm(x))
                        x_normed = self.moe_norm(x_proc).view(-1, D)
                        pipeline_buffers[mb_pre]["normed_input_flat"] = x_normed

                        # 4. Gate (using Normed input)
                        logits = self.gate(x_normed)
                        _, _ = torch.topk(logits, k=TOP_K, dim=1)

                    torch.cuda.nvtx.range_pop()
                ev_pre_ready[mb_pre].record(self.stream_compute)

        torch.cuda.current_stream().wait_stream(self.stream_compute)
        return torch.cat(outputs, dim=0)


# ==========================================
# 2-Block Tiny Lancet Model
# ==========================================
class TinyLancetModel(nn.Module):
    def __init__(self, hidden_dim, num_experts, world_size, group):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Block 1
        self.block1 = LancetBlockConfigurable(
            hidden_dim,
            num_experts,
            world_size,
            group,
            block_name="B1",
            pre_op_module=RealSelfAttention(hidden_dim, 32),
            post_op_module=RealSelfAttention(hidden_dim, 32),
        )

        # Block 2
        self.block2 = LancetBlockConfigurable(
            hidden_dim,
            num_experts,
            world_size,
            group,
            block_name="B2",
            pre_op_module=None,
            post_op_module=nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


# ==========================================
# Worker
# ==========================================
def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12369"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = TinyLancetModel(
        HIDDEN_DIM, NUM_EXPERTS_PER_GPU * world_size, world_size, dist.group.WORLD
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(BATCH_SIZE, 128, HIDDEN_DIM).cuda()

    def trace_handler(p):
        dist.barrier()
        if rank == 0:
            abs_path = os.path.abspath("lancet_residual_corrected.json")
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
        print("[Rank 0] Profiling with Residuals...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=ACTIVE_STEPS, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
    ) as p:
        for _ in range(1 + 1 + ACTIVE_STEPS):
            loss = model(data).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p.step()

    time.sleep(2)
    dist.barrier()
    dist.destroy_process_group()


def run_experiment():
    world_size = 2
    print(f"Starting processes for {world_size} GPUs...")
    mp.start_processes(
        worker, args=(world_size,), nprocs=world_size, join=True, start_method="fork"
    )


run_experiment()
