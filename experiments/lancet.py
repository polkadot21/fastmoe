import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ==========================================
# Configuration (Matches Lancet Fig 4c)
# ==========================================
BATCH_SIZE = 64  # Smaller batch for 2 MBs
MICRO_BATCHES = 2  # MATCHES THE IMAGE (A2A_0, A2A_1)
HIDDEN_DIM = 4096
NUM_EXPERTS_PER_GPU = 2
TOP_K = 2
WARMUP_STEPS = 3
ACTIVE_STEPS = 3


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Heavy MLP to make compute visible in profiling
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

        # 1. Monolithic Ops (Before Pipeline) - e.g. SelfAttn^L in image
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)  # Simulated Attention

        # 2. Gate (Monolithic) - Gate^L in image
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # 3. Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(self.num_local_experts)])

        # 4. Next Ops (The "Extend Forward" part) - e.g. SA^{L+1} in image
        # These will be micro-batched and overlapped with Combine A2A
        self.next_layer_ops = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Simulate Next Layer QKV
            nn.GELU(),
        )

        # Streams
        self.stream_compute = torch.cuda.Stream()  # Runs NextOps (SA^{L+1})
        self.stream_comm = torch.cuda.Stream()  # Runs A2A Dispatch/Combine
        self.stream_expert = torch.cuda.Stream()  # Runs Experts

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # --- Phase 1: Monolithic Pre-Computation (Matches Image: SelfAttn^L, Gate^L) ---
        # The image shows these happen BEFORE the pipeline starts.
        with record_function("Monolithic_PreOps"):
            # Self Attention (Simulated)
            x_attn = self.attn(self.attn_norm(x_flat)) + x_flat

            # Gating
            logits = self.gate(x_attn)
            scores, indices = torch.topk(logits, k=TOP_K, dim=1)

            # Permutation (Simulated cost)
            # In production: torch.scatter to group by expert
            gated_input = x_attn.clone()

        # --- Phase 2: Lancet Pipeline (2 Micro-batches) ---
        # Split for pipelining
        chunks = gated_input.chunk(MICRO_BATCHES)

        # Pipeline State
        pipeline_buffers = [{} for _ in range(MICRO_BATCHES)]
        outputs = [None] * MICRO_BATCHES

        # Synchronization Events
        # We need to signal when stages complete so the next dependent stage can start
        ev_dispatch_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_expert_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]
        ev_combine_ready = [torch.cuda.Event() for _ in range(MICRO_BATCHES)]

        # The Loop: "Extend Pipeline Forwards" (Fig 4c)
        # Time 0: Dispatch_0
        # Time 1: Dispatch_1 | Expert_0
        # Time 2: Combine_0  | Expert_1
        # Time 3: Combine_1  | NextOps_0 (Overlap!)
        # Time 4: NextOps_1

        # We iterate 'tick' from 0 to MICRO_BATCHES + 3
        # Stage offsets:
        # Dispatch: tick
        # Expert:   tick - 1
        # Combine:  tick - 2
        # NextOps:  tick - 3

        for tick in range(MICRO_BATCHES + 3):
            # ----------------------------------------------------------------
            # Stage 4: Next Ops (Compute Stream) - "Extend Pipeline Forwards"
            # Overlaps with Stage 3 (Combine) of the subsequent microbatch
            # ----------------------------------------------------------------
            mb_next = tick - 3
            if 0 <= mb_next < MICRO_BATCHES:
                buf = pipeline_buffers[mb_next]

                # Wait for Combine to finish
                self.stream_compute.wait_event(ev_combine_ready[mb_next])

                with torch.cuda.stream(self.stream_compute):
                    with record_function(f"NextOps_MB{mb_next}"):  # SA^{L+1}
                        # Run the dense ops for the NEXT layer
                        out = self.next_layer_ops(buf["combined_output"])
                        outputs[mb_next] = out

            # ----------------------------------------------------------------
            # Stage 3: Combine All-to-All (Comm Stream)
            # Overlaps with Stage 4 (NextOps) or Stage 2 (Expert)
            # ----------------------------------------------------------------
            mb_comb = tick - 2
            if 0 <= mb_comb < MICRO_BATCHES:
                buf = pipeline_buffers[mb_comb]

                # Wait for Expert Compute to finish
                self.stream_comm.wait_event(ev_expert_ready[mb_comb])

                with torch.cuda.stream(self.stream_comm):
                    with record_function(f"A2A_Combine_MB{mb_comb}"):
                        # Simulated A2A Combine
                        combined = torch.empty_like(buf["dispatch_input"])
                        dist.all_to_all_single(
                            combined, buf["expert_output"], group=self.group, async_op=False
                        )
                        buf["combined_output"] = combined

                # Signal that Combine is done
                ev_combine_ready[mb_comb].record(self.stream_comm)

            # ----------------------------------------------------------------
            # Stage 2: Expert Compute (Expert Stream)
            # Overlaps with Stage 1 (Dispatch) or Stage 3 (Combine)
            # ----------------------------------------------------------------
            mb_exp = tick - 1
            if 0 <= mb_exp < MICRO_BATCHES:
                buf = pipeline_buffers[mb_exp]

                # Wait for Dispatch to finish
                self.stream_expert.wait_event(ev_dispatch_ready[mb_exp])

                with torch.cuda.stream(self.stream_expert):
                    with record_function(f"Expert_MB{mb_exp}"):
                        # Run Experts
                        inp = buf["dispatch_output"]
                        # Simple linear execution of local experts
                        # (Real impl would use grouped GEMM or loop)
                        local_splits = inp.chunk(self.num_local_experts)
                        res = [
                            self.experts[i](local_splits[i]) for i in range(self.num_local_experts)
                        ]
                        buf["expert_output"] = torch.cat(res)

                # Signal that Expert is done
                ev_expert_ready[mb_exp].record(self.stream_expert)

            # ----------------------------------------------------------------
            # Stage 1: Dispatch All-to-All (Comm Stream)
            # ----------------------------------------------------------------
            mb_disp = tick
            if 0 <= mb_disp < MICRO_BATCHES:
                # No wait needed for previous stage (Monolithic is already done on default stream)
                # But we must ensure the Comm stream waits for the default stream to have data ready
                if mb_disp == 0:
                    # Only need to wait once for the monolithic block
                    self.stream_comm.wait_stream(torch.cuda.current_stream())

                with torch.cuda.stream(self.stream_comm):
                    with record_function(f"A2A_Dispatch_MB{mb_disp}"):
                        inp = chunks[mb_disp]
                        pipeline_buffers[mb_disp]["dispatch_input"] = inp

                        out = torch.empty_like(inp)
                        dist.all_to_all_single(out, inp, group=self.group, async_op=False)

                        pipeline_buffers[mb_disp]["dispatch_output"] = out

                # Signal that Dispatch is done
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

    model = LancetBlock(
        HIDDEN_DIM, NUM_EXPERTS_PER_GPU * world_size, world_size, dist.group.WORLD
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(BATCH_SIZE, 128, HIDDEN_DIM).cuda()

    def trace_handler(p):
        if rank == 0:
            p.export_chrome_trace(f"lancet_overlap_mb{MICRO_BATCHES}_rank0.json")
            print("Trace saved. View in Perfetto UI.")

    print(f"Rank {rank}: Warming up...")
    # Warmup
    for _ in range(WARMUP_STEPS):
        loss = model(data).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Rank {rank}: Profiling...")
    with profile(
        activities=[ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=ACTIVE_STEPS),
        on_trace_ready=trace_handler,
    ) as p:
        for _ in range(ACTIVE_STEPS):
            loss = model(data).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
