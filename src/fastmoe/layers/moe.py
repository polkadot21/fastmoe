import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function  # Added import

from fastmoe import consts
from fastmoe.kernels.ops import grouped_weighted_scatter_add


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        ff_dim,
        num_experts=8,
        top_k=2,
        implementation=consts.MoEImplementation.FAST,
        group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.implementation = implementation
        self.group = group

        if dist.is_initialized():
            self.world_size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
        else:
            self.world_size = 1
            self.rank = 0

        assert num_experts % self.world_size == 0, "Experts must divide evenly"
        self.num_local_experts = num_experts // self.world_size

        self.router = nn.Linear(dim, num_experts, bias=False)
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

    def gate_and_sort(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1: Local Gating"""
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        logits = self.router(x_flat)
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)

        indices_flat = topk_indices.view(-1)
        weights_flat = topk_weights.view(-1)

        sort_indices = torch.argsort(indices_flat)
        expert_indices_sorted = indices_flat[sort_indices]

        src_indices = torch.arange(x_flat.shape[0], device=x.device).repeat_interleave(self.top_k)
        reverse_map_indices = src_indices[sort_indices]
        sorted_weights = weights_flat[sort_indices]

        permuted_data = x_flat[reverse_map_indices]

        expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)
        expert_counts = expert_counts.to(dtype=torch.long)

        return permuted_data, expert_counts, reverse_map_indices, sorted_weights, (B, T)

    def dispatch_exchange_async(self, permuted_data, local_expert_counts, tag="Dispatch"):
        """
        Stage 2: Async Communication
        """
        if self.world_size == 1:
            return permuted_data, None, (local_expert_counts.tolist(), local_expert_counts.tolist())

        # 1. Exchange Metadata (Blocking)
        send_counts_per_rank = local_expert_counts.view(self.world_size, self.num_local_experts)
        recv_counts_per_rank = torch.empty_like(send_counts_per_rank)

        # We can also tag this metadata exchange if desired, but it's usually fast
        dist.all_to_all_single(recv_counts_per_rank, send_counts_per_rank, group=self.group)

        recv_counts_list = recv_counts_per_rank.view(-1).tolist()

        send_splits = send_counts_per_rank.sum(dim=1).tolist()
        recv_splits = recv_counts_per_rank.sum(dim=1).tolist()

        total_recv = sum(recv_splits)
        recv_data = torch.empty(
            total_recv, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )

        # 2. Async Data Exchange
        # Wrap the specific NCCL call with the tag
        with record_function(f"NCCL: {tag}"):
            handle = dist.all_to_all_single(
                recv_data,
                permuted_data,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.group,
                async_op=True,
            )

        return recv_data, handle, (recv_counts_list, send_splits)

    def compute_experts(self, recv_data, flat_counts_list) -> torch.Tensor:
        """Stage 3: Batched Expert Computation"""
        results_by_rank = [[] for _ in range(self.world_size)]
        current_offset = 0
        list_idx = 0
        expert_slices = [[] for _ in range(self.num_local_experts)]

        for rank in range(self.world_size):
            for local_exp in range(self.num_local_experts):
                count = flat_counts_list[list_idx]
                list_idx += 1
                if count > 0:
                    expert_slices[local_exp].append((current_offset, current_offset + count, rank))
                    current_offset += count

        for i, slices in enumerate(expert_slices):
            if not slices:
                continue
            batch_chunks = [recv_data[s:e] for s, e, r in slices]
            expert_input = torch.cat(batch_chunks, dim=0)
            expert_output = self.experts[i](expert_input)
            chunk_sizes = [e - s for s, e, r in slices]
            splitted_out = expert_output.split(chunk_sizes)
            for j, (_, __, rank) in enumerate(slices):
                results_by_rank[rank].append(splitted_out[j])

        final_chunks = []
        for rank_list in results_by_rank:
            if rank_list:
                final_chunks.append(torch.cat(rank_list, dim=0))

        if not final_chunks:
            return torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)

        return torch.cat(final_chunks, dim=0)

    def combine_exchange_async(self, expert_output, recv_counts_list, send_splits, tag="Combine"):
        """Stage 4: Reverse Async Communication"""
        if self.world_size == 1:
            return expert_output, None

        output_splits = []
        offset = 0
        for _ in range(self.world_size):
            rank_sum = sum(recv_counts_list[offset : offset + self.num_local_experts])
            output_splits.append(rank_sum)
            offset += self.num_local_experts

        input_splits = send_splits
        total_back = sum(input_splits)
        final_data = torch.empty(
            total_back, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )

        with record_function(f"NCCL: {tag}"):
            handle = dist.all_to_all_single(
                final_data,
                expert_output,
                output_split_sizes=input_splits,
                input_split_sizes=output_splits,
                group=self.group,
                async_op=True,
            )
        return final_data, handle

    def unpermute(self, x_out, reverse_map_indices, sorted_weights, original_shape) -> torch.Tensor:
        """Stage 5: Local Un-permute"""
        B, T = original_shape
        D = self.dim
        if self.implementation == consts.MoEImplementation.STANDARD:
            x_out = x_out * sorted_weights.unsqueeze(-1)
            out = torch.zeros(B * T, D, device=x_out.device, dtype=x_out.dtype)
            out.index_add_(0, reverse_map_indices, x_out)
            return out.view(B, T, D)
        elif self.implementation == consts.MoEImplementation.FAST:
            out = grouped_weighted_scatter_add(
                [x_out], reverse_map_indices, sorted_weights, (B * T, D)
            )
            return out.view(B, T, D)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Forward Pass"""
        permuted, expert_counts, rev_idx, weights, shape = self.gate_and_sort(x)
        # Pass standard tags for synchronous execution
        recv_data, handle, (recv_counts, send_splits) = self.dispatch_exchange_async(
            permuted, expert_counts, tag="Sync_Dispatch"
        )
        if handle:
            handle.wait()

        expert_out = self.compute_experts(recv_data, recv_counts)

        final_data, handle = self.combine_exchange_async(
            expert_out, recv_counts, send_splits, tag="Sync_Combine"
        )
        if handle:
            handle.wait()

        return self.unpermute(final_data, rev_idx, weights, shape)
