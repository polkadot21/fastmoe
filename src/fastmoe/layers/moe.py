import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from fastmoe import consts
from fastmoe.distributed import Communicator, PytorchCommunicator
from fastmoe.kernels.ops import weighted_scatter_add


def _calibrate_cycles_per_ms(device=None, target_ms=5.0):
    if device is None:
        device = torch.device("cuda")
    torch.cuda.synchronize(device=device)
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    cycles = int(5e6)
    for _ in range(12):
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
        torch.cuda.synchronize(device=device)
        ms = start.elapsed_time(end)
        if ms <= 0:
            cycles *= 2
            continue
        scale = target_ms / ms
        cycles = int(cycles * (0.5 + 0.5 * scale))
        if abs(ms - target_ms) / target_ms < 0.05:
            break
    return cycles / target_ms  # cycles per ms


# Cache per process
_CYCLES_PER_MS = None


def gpu_sleep_ms(ms: float, device: torch.device):
    global _CYCLES_PER_MS
    if ms <= 0:
        return
    if _CYCLES_PER_MS is None:
        _CYCLES_PER_MS = _calibrate_cycles_per_ms(device=device, target_ms=5.0)
    torch.cuda._sleep(int(ms * _CYCLES_PER_MS))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        ff_dim,
        num_experts=8,
        top_k=2,
        implementation=consts.MoEImplementation.FAST,
        communicator: Communicator = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.implementation = implementation
        self.comm = communicator if communicator else PytorchCommunicator()

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        assert num_experts % self.world_size == 0, "Total experts must divide evenly by world size"
        self.num_local_experts = num_experts // self.world_size

        self.local_expert_indices = [
            self.rank * self.num_local_experts + i for i in range(self.num_local_experts)
        ]

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim, bias=False),
                )
                for _ in range(self.num_local_experts)
            ]
        )

        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        group=None,
        comm_stream=None,
        simulate_a2a_ms: float = 30.0,
    ) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [BT, D]
        n_tokens = x_flat.shape[0]

        # ---------------------------------------------------------------------
        # 1. Routing
        # ---------------------------------------------------------------------
        logits = self.router(x_flat)
        router_probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)  # [BT, K] each

        # Flatten routing decisions
        expert_idx_flat = indices.reshape(-1)  # [BT*K]
        w_flat = weights.reshape(-1).to(x_flat.dtype)  # match expert dtype for kernel

        # For recombination: token id for each of the K routes
        tok_flat = torch.arange(n_tokens, device=x.device, dtype=torch.long).repeat_interleave(
            self.top_k
        )  # [BT*K]

        # ---------------------------------------------------------------------
        # 2. Metadata: Histogram (per rank / per local expert), exchanged on SAME lane
        # ---------------------------------------------------------------------
        global_expert_hist = torch.bincount(expert_idx_flat, minlength=self.num_experts).int()
        expert_hist_per_rank = global_expert_hist.view(self.world_size, self.num_local_experts)
        send_counts = expert_hist_per_rank.sum(dim=1).long()

        if self.world_size == 1:
            recv_expert_hist = expert_hist_per_rank
        else:
            recv_expert_hist = torch.empty_like(expert_hist_per_rank)
            dist.all_to_all_single(recv_expert_hist, expert_hist_per_rank, group=group)

        recv_counts = recv_expert_hist.sum(dim=1).long()

        # ---------------------------------------------------------------------
        # 3. Permute & Dispatch (Heavy Comm)
        # ---------------------------------------------------------------------
        # Sort by destination expert id -> contiguous by rank then expert (given your mapping)
        sorted_indices = torch.argsort(expert_idx_flat)
        x_rep = x_flat.repeat_interleave(self.top_k, dim=0)  # [BT*K, D]
        x_permuted = x_rep.index_select(0, sorted_indices)

        # Simulate network delay on comm stream if specified
        recv_data = self.all_to_all(
            x_permuted,
            send_counts.tolist(),
            recv_counts.tolist(),
            group=group,
            comm_stream=comm_stream,
            simulate_ms=simulate_a2a_ms,
        )

        # ---------------------------------------------------------------------
        # 4. Computation (Experts)
        # ---------------------------------------------------------------------
        chunks = recv_data.split(recv_expert_hist.reshape(-1).tolist())

        expert_outputs = []
        for i in range(self.num_local_experts):
            current_expert_chunks = []
            for r in range(self.world_size):
                chunk_idx = r * self.num_local_experts + i
                current_expert_chunks.append(chunks[chunk_idx])

            expert_input = (
                torch.cat(current_expert_chunks, dim=0) if current_expert_chunks else None
            )

            if expert_input is not None and expert_input.numel() > 0:
                out = self.experts[i](expert_input)
            else:
                out = torch.empty(0, self.dim, device=x.device, dtype=x.dtype)

            split_sizes = recv_expert_hist[:, i].tolist()
            if sum(split_sizes) > 0:
                split_out = out.split(split_sizes)
                expert_outputs.append(split_out)
            else:
                expert_outputs.append(
                    [torch.empty(0, self.dim, device=x.device, dtype=x.dtype)] * self.world_size
                )

        # ---------------------------------------------------------------------
        # 5. Reassemble by destination rank & return all-to-all
        # ---------------------------------------------------------------------
        to_send_back = []
        for r in range(self.world_size):
            rank_chunks = []
            for i in range(self.num_local_experts):
                rank_chunks.append(expert_outputs[i][r])

            if rank_chunks:
                to_send_back.append(torch.cat(rank_chunks, dim=0))
            else:
                to_send_back.append(torch.empty(0, self.dim, device=x.device, dtype=x.dtype))

        final_out_buffer = (
            torch.cat(to_send_back, dim=0)
            if to_send_back
            else torch.empty(0, self.dim, device=x.device, dtype=x.dtype)
        )

        # Simulate network delay on comm stream if specified for return
        combined_output = self.all_to_all(
            final_out_buffer,
            recv_counts.tolist(),
            send_counts.tolist(),
            group=group,
            comm_stream=comm_stream,
            simulate_ms=simulate_a2a_ms,
        )

        # ---------------------------------------------------------------------
        # 6. FAST recombination: scatter-add directly into [BT, D]
        # ---------------------------------------------------------------------
        # `combined_output` aligns with the `sorted_indices` order (same ordering you used to send).
        w_perm = w_flat.index_select(0, sorted_indices)
        tok_perm = tok_flat.index_select(0, sorted_indices)

        out_flat = weighted_scatter_add(combined_output, tok_perm, w_perm, out_shape=(n_tokens, D))
        return out_flat.view(B, T, D)

    def all_to_all(
        self, x, send_counts, recv_counts, group=None, comm_stream=None, simulate_ms: float = 0.0
    ):
        if self.world_size == 1:
            return x

        total_recv = sum(recv_counts)
        output = torch.empty(total_recv, x.size(1), device=x.device, dtype=x.dtype)

        # Default behavior is to run on current stream
        if comm_stream is None:
            if simulate_ms > 0:
                gpu_sleep_ms(simulate_ms, x.device)
            dist.all_to_all_single(
                output,
                x,
                output_split_sizes=recv_counts,
                input_split_sizes=send_counts,
                group=group,
            )
            return output

        # Run NCCL on comm stream
        with torch.cuda.stream(comm_stream):
            # Ensure allocator doesn't reuse buffers too early
            x.record_stream(comm_stream)
            output.record_stream(comm_stream)

            if simulate_ms > 0:
                gpu_sleep_ms(simulate_ms, x.device)

            dist.all_to_all_single(
                output,
                x,
                output_split_sizes=recv_counts,
                input_split_sizes=send_counts,
                group=group,
            )

        # Consumer is current stream, so it must wait for comm_stream to finish
        torch.cuda.current_stream().wait_stream(comm_stream)
        return output
