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
MICRO_BATCHES = 2
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
        return out + residual


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# Flexible Lancet Block
# ==========================================
class LancetBlockConfigurable(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_experts,
        world_size,
        group,
        layer_id,
        use_pre_attn=True,
        post_op_module=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.use_pre_attn = use_pre_attn
        self.hidden_dim = hidden_dim
        self.num_local_experts = num_experts // world_size
        self.group = group
        self.world_size = world_size

        # 1. Pre-Ops
        if self.use_pre_attn:
            self.attn = RealSelfAttention(hidden_dim, 32)

        # Gating always runs
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # 2. Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # 3. Post-Ops (Passed from outside to allow chaining/Output layer)
        self.post_ops = post_op_module if post_op_module else nn.Identity()

        # Streams
        self.stream_compute = torch.cuda.Stream()
        self.stream_comm = torch.cuda.Stream()
        self.stream_expert = torch.cuda.Stream()

    def forward(self, x):
        # x shape: [Batch, Seq, Dim]
        B, S, D = x.shape
        chunks = x.chunk(MICRO_BATCHES, dim=0)

        pipeline_buffers = [{} for _ in range(MICRO_BATCHES)]
        outputs = [None] * MICRO_BATCHES

        # Events
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
                    label = f"L{self.layer_id}_PostOps_MB{mb_post}"
                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        # Reshape for Attention/Linear
                        flat_out = buf["combined_output"]
                        mb_size = chunks[mb_post].size(0)
                        reshaped_in = flat_out.view(mb_size, S, D)
                        outputs[mb_post] = self.post_ops(reshaped_in)
                    torch.cuda.nvtx.range_pop()

            # --- Stage 4: Combine A2A (Comm Stream) ---
            mb_comb = tick - 3
            if 0 <= mb_comb < MICRO_BATCHES:
                buf = pipeline_buffers[mb_comb]
                self.stream_comm.wait_event(ev_expert_ready[mb_comb])
                with torch.cuda.stream(self.stream_comm):
                    label = f"L{self.layer_id}_Combine_MB{mb_comb}"
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
                    label = f"L{self.layer_id}_Expert_MB{mb_exp}"
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
                    label = f"L{self.layer_id}_Dispatch_MB{mb_disp}"
                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        real_in = buf["gated_input"]
                        bloated_in = real_in.repeat(1, COMM_SCALING_FACTOR)
                        bloated_out = torch.empty_like(bloated_in)
                        dist.all_to_all_single(
                            bloated_out, bloated_in, group=self.group, async_op=False
                        )
                        pipeline_buffers[mb_disp]["dispatch_output"] = torch.empty_like(real_in)
                    torch.cuda.nvtx.range_pop()
                ev_dispatch_ready[mb_disp].record(self.stream_comm)

            # --- Stage 1: Pre-Ops (Compute Stream) ---
            mb_pre = tick
            if 0 <= mb_pre < MICRO_BATCHES:
                with torch.cuda.stream(self.stream_compute):
                    label = f"L{self.layer_id}_PreOps_MB{mb_pre}"
                    torch.cuda.nvtx.range_push(label)
                    with record_function(label):
                        x_chunk = chunks[mb_pre]

                        # Conditional Attention (Skip if Block 2)
                        if self.use_pre_attn:
                            x_proc = self.attn(x_chunk)
                        else:
                            x_proc = x_chunk  # Already Attended

                        x_flat = x_proc.view(-1, D)
                        logits = self.gate(x_flat)
                        _, _ = torch.topk(logits, k=TOP_K, dim=1)

                        pipeline_buffers[mb_pre]["gated_input"] = x_flat.clone()
                    torch.cuda.nvtx.range_pop()
                ev_pre_ready[mb_pre].record(self.stream_compute)

        torch.cuda.current_stream().wait_stream(self.stream_compute)
        return torch.cat(outputs, dim=0)


# ==========================================
# Tiny Model Container
# ==========================================
class TinyLancetModel(nn.Module):
    def __init__(self, hidden_dim, num_experts, world_size, group):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)  # Linear Input Layer

        # Block 1: Computes Attn L (Pre) and Attn L+1 (Post)
        # Post-Op is RealSelfAttention (Attn L+1)
        self.block1 = LancetBlockConfigurable(
            hidden_dim,
            num_experts,
            world_size,
            group,
            layer_id=1,
            use_pre_attn=True,
            post_op_module=RealSelfAttention(hidden_dim, 32),
        )

        # Block 2: Takes Attn L+1 Output. Skips Pre-Attn.
        # Post-Op is Linear Output (to hide Combine)
        self.block2 = LancetBlockConfigurable(
            hidden_dim,
            num_experts,
            world_size,
            group,
            layer_id=2,
            use_pre_attn=False,  # Input is already attended!
            post_op_module=nn.Linear(hidden_dim, hidden_dim),  # Linear Output Block
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.block1(x)  # Returns Attn(L+1) output
        x = self.block2(x)  # Returns Linear Output
        return x


# ==========================================
# Worker
# ==========================================
def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12367"
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
            abs_path = os.path.abspath("lancet_tiny_model_chain.json")
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
        print("[Rank 0] Profiling 2-Block Chain...")
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
