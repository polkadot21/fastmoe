import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ==========================================
# Configuration (Matches Lancet Fig 4c)
# ==========================================
BATCH_SIZE = 64  # Small batch for 2 MBs (32 per MB)
MICRO_BATCHES = 2  # Strictly 2 micro-batches
HIDDEN_DIM = 4096  # Large enough to make compute visible
NUM_EXPERTS_PER_GPU = 2
TOP_K = 2
WARMUP_STEPS = 5
ACTIVE_STEPS = 5


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Heavy MLP to make compute visible in profiling vs communication
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LancetBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts, world_size, group):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_local_experts = num_experts // world_size
        self.world_size = world_size
        self.group = group

        # 1. Monolithic Ops (Before Pipeline) - e.g. SelfAttn^L
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)

        # 2. Gate (Monolithic) - Gate^L
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # 3. Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # 4. Next Ops (The "Extend Forward" part) - e.g. SA^{L+1}
        # These will be micro-batched and overlapped with Combine A2A
        self.next_layer_ops = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

        # Streams
        self.stream_compute = torch.cuda.Stream()  # Runs NextOps (SA^{L+1})
        self.stream_comm = torch.cuda.Stream()  # Runs A2A Dispatch/Combine
        self.stream_expert = torch.cuda.Stream()  # Runs Experts

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # --- Phase 1: Monolithic Pre-Computation ---
        with record_function("Monolithic_PreOps"):
            # Self Attention (Simulated)
            x_attn = self.attn(self.attn_norm(x_flat)) + x_flat

            # Gating
            logits = self.gate(x_attn)
            _, indices = torch.topk(logits, k=TOP_K, dim=1)

            # Permutation (Simulated cost)
            gated_input = x_attn.clone()

        # --- Phase 2: Lancet Pipeline (2 Micro-batches) ---
        chunks = gated_input.chunk(MICRO_BATCHES)

        # Pipeline State
        pipeline_buffers = [{} for _ in range(MICRO_BATCHES)]
        outputs = [None] * MICRO_BATCHES

        # Synchronization Events
        ev_dispatch_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_expert_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_combine_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]

        # Tick 0: Dispatch_0
        # Tick 1: Dispatch_1 | Expert_0
        # Tick 2: Combine_0  | Expert_1
        # Tick 3: Combine_1  | NextOps_0 (Overlap!)
        # Tick 4: NextOps_1

        for tick in range(MICRO_BATCHES + 3):
            # Stage 4: Next Ops (Compute Stream)
            mb_next = tick - 3
            if 0 <= mb_next < MICRO_BATCHES:
                buf = pipeline_buffers[mb_next]
                self.stream_compute.wait_event(ev_combine_ready[mb_next])

                with torch.cuda.stream(self.stream_compute):
                    with record_function(f"NextOps_MB{mb_next}"):
                        out = self.next_layer_ops(buf["combined_output"])
                        outputs[mb_next] = out

            # Stage 3: Combine All-to-All (Comm Stream)
            mb_comb = tick - 2
            if 0 <= mb_comb < MICRO_BATCHES:
                buf = pipeline_buffers[mb_comb]
                self.stream_comm.wait_event(ev_expert_ready[mb_comb])

                with torch.cuda.stream(self.stream_comm):
                    with record_function(f"A2A_Combine_MB{mb_comb}"):
                        combined = torch.empty_like(buf["dispatch_input"])
                        dist.all_to_all_single(
                            combined, buf["expert_output"], group=self.group, async_op=False
                        )
                        buf["combined_output"] = combined
                ev_combine_ready[mb_comb].record(self.stream_comm)

            # Stage 2: Expert Compute (Expert Stream)
            mb_exp = tick - 1
            if 0 <= mb_exp < MICRO_BATCHES:
                buf = pipeline_buffers[mb_exp]
                self.stream_expert.wait_event(ev_dispatch_ready[mb_exp])

                with torch.cuda.stream(self.stream_expert):
                    with record_function(f"Expert_MB{mb_exp}"):
                        inp = buf["dispatch_output"]
                        local_splits = inp.chunk(self.num_local_experts)
                        res = [
                            self.experts[i](local_splits[i]) for i in range(self.num_local_experts)
                        ]
                        buf["expert_output"] = torch.cat(res)
                ev_expert_ready[mb_exp].record(self.stream_expert)

            # Stage 1: Dispatch All-to-All (Comm Stream)
            mb_disp = tick
            if 0 <= mb_disp < MICRO_BATCHES:
                if mb_disp == 0:
                    self.stream_comm.wait_stream(torch.cuda.current_stream())

                with torch.cuda.stream(self.stream_comm):
                    with record_function(f"A2A_Dispatch_MB{mb_disp}"):
                        inp = chunks[mb_disp]
                        pipeline_buffers[mb_disp]["dispatch_input"] = inp
                        out = torch.empty_like(inp)
                        dist.all_to_all_single(out, inp, group=self.group, async_op=False)
                        pipeline_buffers[mb_disp]["dispatch_output"] = out
                ev_dispatch_ready[mb_disp].record(self.stream_comm)

        # Final Sync
        torch.cuda.current_stream().wait_stream(self.stream_compute)
        final_out = torch.cat(outputs, dim=0)
        return final_out.view(B, S, D)


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Model Setup
    model = LancetBlock(
        HIDDEN_DIM, NUM_EXPERTS_PER_GPU * world_size, world_size, dist.group.WORLD
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(BATCH_SIZE, 128, HIDDEN_DIM).cuda()

    # ------------------------------------------------------------
    # Profiler Handler (Rank 0 only)
    # ------------------------------------------------------------
    def trace_handler(p):
        if rank == 0:
            # Add metadata to the trace
            p.add_metadata("World Size", str(world_size))
            p.add_metadata("Batch Size", str(BATCH_SIZE))
            p.add_metadata("Micro Batches", str(MICRO_BATCHES))
            p.add_metadata("Hidden Dim", str(HIDDEN_DIM))

            output_file = "lancet_overlap_rank0.json"
            p.export_chrome_trace(output_file)
            print("\n[Rank 0] Profiling Complete.")
            print(f"[Rank 0] Trace saved to: {output_file}")
            print(f"[Rank 0] Metadata: GPUs={world_size}, Batch={BATCH_SIZE}, MB={MICRO_BATCHES}")

    if rank == 0:
        print(f"[Rank 0] Starting Warmup ({WARMUP_STEPS} steps)...")

    # Warmup
    for _ in range(WARMUP_STEPS):
        loss = model(data).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if rank == 0:
        print(f"[Rank 0] Starting Active Profiling ({ACTIVE_STEPS} steps)...")

    # ------------------------------------------------------------
    # Profiling Loop
    # Added ProfilerActivity.CPU to capture 'record_function' labels
    # ------------------------------------------------------------
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=ACTIVE_STEPS, repeat=1),
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


n_gpus = torch.cuda.device_count()
world_size = 8

if n_gpus < world_size:
    print(f"Error: Script requires at least {world_size} GPUs, but found {n_gpus}.")
else:
    print(f"Launching Lancet Demo on {world_size} GPUs...")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
