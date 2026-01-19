import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 128
MICRO_BATCHES = 4
HIDDEN_DIM = 4096
NUM_EXPERTS_PER_GPU = 2
TOP_K = 2
WARMUP_STEPS = 5
ACTIVE_STEPS = 3


# ==========================================
# Model Definition
# ==========================================
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

        # 1. Pre-Ops
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # 2. Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # 3. Post-Ops
        self.next_layer_ops = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

        # Streams
        self.stream_compute = torch.cuda.Stream()
        self.stream_comm = torch.cuda.Stream()
        self.stream_expert = torch.cuda.Stream()

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)
        chunks = x_flat.chunk(MICRO_BATCHES)

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
                    torch.cuda.nvtx.range_push(f"PostOps_MB{mb_post}")
                    with record_function(f"PostOps_MB{mb_post}"):
                        out = self.next_layer_ops(buf["combined_output"])
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
                        combined = torch.empty_like(buf["dispatch_input"])
                        dist.all_to_all_single(
                            combined, buf["expert_output"], group=self.group, async_op=False
                        )
                        buf["combined_output"] = combined
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
                        inp = buf["gated_input"]
                        pipeline_buffers[mb_disp]["dispatch_input"] = inp
                        out = torch.empty_like(inp)
                        dist.all_to_all_single(out, inp, group=self.group, async_op=False)
                        pipeline_buffers[mb_disp]["dispatch_output"] = out
                    torch.cuda.nvtx.range_pop()
                ev_dispatch_ready[mb_disp].record(self.stream_comm)

            # --- Stage 1: Pre-Ops (Compute Stream) ---
            mb_pre = tick
            if 0 <= mb_pre < MICRO_BATCHES:
                with torch.cuda.stream(self.stream_compute):
                    torch.cuda.nvtx.range_push(f"PreOps_MB{mb_pre}")
                    with record_function(f"PreOps_MB{mb_pre}"):
                        x_chunk = chunks[mb_pre]
                        x_attn = self.attn(self.attn_norm(x_chunk)) + x_chunk
                        logits = self.gate(x_attn)
                        _, indices = torch.topk(logits, k=TOP_K, dim=1)
                        gated = x_attn.clone()
                        pipeline_buffers[mb_pre]["gated_input"] = gated
                    torch.cuda.nvtx.range_pop()
                ev_pre_ready[mb_pre].record(self.stream_compute)

        torch.cuda.current_stream().wait_stream(self.stream_compute)
        final_out = torch.cat(outputs, dim=0)
        return final_out.view(B, S, D)


# ==========================================
# Worker Function
# ==========================================
def worker(rank, world_size):
    # Set up distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"  # New port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Initialize Model
    model = LancetBlockFull(
        HIDDEN_DIM, NUM_EXPERTS_PER_GPU * world_size, world_size, dist.group.WORLD
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(BATCH_SIZE, 128, HIDDEN_DIM).cuda()

    # Trace Handler
    def trace_handler(p):
        dist.barrier()
        if rank == 0:
            p.add_metadata("Config", f"MB={MICRO_BATCHES}, GPUs={world_size}")
            abs_path = os.path.abspath("lancet_notebook_trace_rank0.json")
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

    # Profiling
    if rank == 0:
        print("[Rank 0] Profiling...")
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

    # Cleanup
    if rank == 0:
        print("[Rank 0] Flushing traces...")
    time.sleep(2)
    dist.barrier()
    dist.destroy_process_group()


# ==========================================
# Execution (Notebook Cell Friendly)
# ==========================================
def run_experiment():
    # Hardcoded to 2 for safety in notebook execution; set to 8 if you have 8 GPUs
    world_size = 2

    print(f"Starting processes with method='fork' for {world_size} GPUs...")

    # Use mp.start_processes with 'fork'
    # NOTE: This works in a notebook cell because 'fork' copies the memory,
    # so 'worker' function is visible to children without being importable.
    mp.start_processes(
        worker, args=(world_size,), nprocs=world_size, join=True, start_method="fork"
    )


# Run it directly
run_experiment()
