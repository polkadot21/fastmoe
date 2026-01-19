import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ==========================================
# Configuration (Matches Lancet Fig 4d: Full Overlap)
# ==========================================
BATCH_SIZE = 128
MICRO_BATCHES = 2  # Using 4 to clearly see the 5-stage pipeline depth
HIDDEN_DIM = 4096
NUM_EXPERTS_PER_GPU = 2
TOP_K = 2
WARMUP_STEPS = 5
ACTIVE_STEPS = 5


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

        # 1. Pre-Ops (Current Layer Attention + Gate)
        # In Fig 4d, these are ALSO micro-batched
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # 2. Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # 3. Post-Ops (Next Layer Attention)
        self.next_layer_ops = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

        # Streams
        self.stream_compute = torch.cuda.Stream()  # Runs Pre-Ops AND Post-Ops
        self.stream_comm = torch.cuda.Stream()  # Runs A2A Dispatch/Combine
        self.stream_expert = torch.cuda.Stream()  # Runs Experts

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Split Raw Input for Pre-Ops Micro-batching (Fig 4d)
        chunks = x_flat.chunk(MICRO_BATCHES)

        pipeline_buffers = [{} for _ in range(MICRO_BATCHES)]
        outputs = [None] * MICRO_BATCHES

        # Events for dependency management
        ev_pre_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_dispatch_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_expert_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_combine_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]

        # ----------------------------------------------------------------
        # 5-Stage Pipeline Loop (Pre -> Disp -> Exp -> Comb -> Post)
        # ----------------------------------------------------------------
        # We iterate enough ticks to flush the pipeline.
        # Tick offsets:
        # PreOps:   tick
        # Dispatch: tick - 1
        # Expert:   tick - 2
        # Combine:  tick - 3
        # PostOps:  tick - 4

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
                        pipeline_buffers[mb_disp]["dispatch_input"] = inp  # Save for shape ref
                        out = torch.empty_like(inp)
                        dist.all_to_all_single(out, inp, group=self.group, async_op=False)
                        pipeline_buffers[mb_disp]["dispatch_output"] = out
                    torch.cuda.nvtx.range_pop()
                ev_dispatch_ready[mb_disp].record(self.stream_comm)

            # --- Stage 1: Pre-Ops (Compute Stream) - Fig 4d Split ---
            mb_pre = tick
            if 0 <= mb_pre < MICRO_BATCHES:
                # Pre-ops run on Compute stream.
                # Note: Post-ops ALSO run on Compute stream (Stage 5).
                # Since 'tick' loop is serial on CPU, Stage 5 (Post) was launched first
                # into the Compute stream above. Stage 1 (Pre) is queued AFTER it.
                # This naturally serializes Pre/Post ops on the same stream, which is fine/desired.

                with torch.cuda.stream(self.stream_compute):
                    torch.cuda.nvtx.range_push(f"PreOps_MB{mb_pre}")
                    with record_function(f"PreOps_MB{mb_pre}"):
                        x_chunk = chunks[mb_pre]
                        # Attention + Gate
                        x_attn = self.attn(self.attn_norm(x_chunk)) + x_chunk
                        logits = self.gate(x_attn)
                        _, indices = torch.topk(logits, k=TOP_K, dim=1)
                        # Simulate Permutation
                        gated = x_attn.clone()
                        pipeline_buffers[mb_pre]["gated_input"] = gated
                    torch.cuda.nvtx.range_pop()
                ev_pre_ready[mb_pre].record(self.stream_compute)

        # Final Sync
        torch.cuda.current_stream().wait_stream(self.stream_compute)
        final_out = torch.cat(outputs, dim=0)
        return final_out.view(B, S, D)


def worker(rank, world_size):
    # Unique port to avoid conflicts if re-running
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Setup
    model = LancetBlockFull(
        HIDDEN_DIM, NUM_EXPERTS_PER_GPU * world_size, world_size, dist.group.WORLD
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(BATCH_SIZE, 128, HIDDEN_DIM).cuda()

    # ------------------------------------------------------------
    # Trace Handler (Strictly Rank 0)
    # ------------------------------------------------------------
    def trace_handler(p):
        if rank == 0:
            p.add_metadata("Config", f"MB={MICRO_BATCHES}, GPUs={world_size}")
            # Export trace
            p.export_chrome_trace("lancet_full_fig4d_rank0.json")
            print("[Rank 0] Trace saved: lancet_full_fig4d_rank0.json")

    # Warmup
    if rank == 0:
        print("[Rank 0] Warming up...")
    for _ in range(WARMUP_STEPS):
        loss = model(data).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Profile
    if rank == 0:
        print("[Rank 0] Profiling...")
    # Included CPU activity to capture the labels
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=ACTIVE_STEPS),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
    ) as p:
        for _ in range(ACTIVE_STEPS):
            loss = model(data).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = 2

    if n_gpus < world_size:
        print(f"Error: Need {world_size} GPUs, found {n_gpus}")
    else:
        print(f"Launching Lancet Full Pipeline (Fig 4d) on {world_size} GPUs")
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
