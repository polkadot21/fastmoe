import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

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
        """Stage 1: Local Gating and Indices Calculation"""
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # Router
        logits = self.router(x_flat)
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)

        # Flatten for routing
        indices_flat = topk_indices.view(-1)
        weights_flat = topk_weights.view(-1)

        # Sort tokens by expert ID to make memory access contiguous
        sort_indices = torch.argsort(indices_flat)
        expert_indices_sorted = indices_flat[sort_indices]

        # Mapping to restore original order later
        src_indices = torch.arange(x_flat.shape[0], device=x.device).repeat_interleave(self.top_k)
        reverse_map_indices = src_indices[sort_indices]
        sorted_weights = weights_flat[sort_indices]

        # Permute input data to be grouped by expert
        permuted_data = x_flat[reverse_map_indices]

        # Count tokens per expert [Num_Experts]
        # This is critical: we need the count per expert to know how to slice the received buffer
        expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)
        expert_counts = expert_counts.to(dtype=torch.long)

        return permuted_data, expert_counts, reverse_map_indices, sorted_weights, (B, T)

    def dispatch_exchange_async(self, permuted_data, local_expert_counts):
        """
        Stage 2: Async Communication (Dispatch)
        Returns: (recv_data, handle, (recv_counts_per_rank, send_splits))
        """
        if self.world_size == 1:
            # Fake handle for single GPU
            return permuted_data, None, (local_expert_counts, local_expert_counts)

        # 1. Exchange Metadata (Blocking)
        # We perform a blocking exchange of counts first to size the buffers.
        # local_expert_counts is [Num_Experts]. Reshape to [World_Size, Local_Experts_Per_Rank]
        send_counts_per_rank = local_expert_counts.view(self.world_size, self.num_local_experts)
        recv_counts_per_rank = torch.empty_like(send_counts_per_rank)

        # This is lightweight (integers only)
        dist.all_to_all_single(recv_counts_per_rank, send_counts_per_rank, group=self.group)

        send_splits = send_counts_per_rank.sum(dim=1).tolist()
        recv_splits = recv_counts_per_rank.sum(dim=1).tolist()

        total_recv = sum(recv_splits)
        recv_data = torch.empty(
            total_recv, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )

        # 2. Async Data Exchange
        # Returns a Work handle. CPU continues immediately.
        handle = dist.all_to_all_single(
            recv_data,
            permuted_data,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
            async_op=True,
        )

        # Return data + handle + metadata needed for next steps
        return recv_data, handle, (recv_counts_per_rank, send_splits)

    def compute_experts(self, recv_data, counts) -> torch.Tensor:
        """
        Stage 3: Batched Expert Computation
        Args:
            recv_data: The buffer received from All-to-All
            counts: [World_Size, Num_Local_Experts] matrix of token counts
        """
        # Flat counts to iterate through the linear recv_data buffer
        flat_counts = counts.view(-1)
        offsets = flat_counts.cumsum(0)
        start_offsets = torch.cat(
            (torch.zeros(1, device=recv_data.device, dtype=torch.long), offsets[:-1])
        )
        end_offsets = offsets

        # We need to organize results by SOURCE rank for the return trip
        results_by_rank = [[] for _ in range(self.world_size)]

        for local_expert_idx in range(self.num_local_experts):
            # 1. Gather all tokens for this expert from ALL ranks
            batch_chunks = []
            chunk_sizes = []

            for rank in range(self.world_size):
                # Calculate index in the flat buffer
                # Layout is Rank-Major: Rank0_Exp0, Rank0_Exp1, ... Rank1_Exp0 ...
                flat_idx = rank * self.num_local_experts + local_expert_idx

                size = flat_counts[flat_idx].item()
                chunk_sizes.append(size)

                if size > 0:
                    start = start_offsets[flat_idx].item()
                    end = end_offsets[flat_idx].item()
                    batch_chunks.append(recv_data[start:end])

            if not batch_chunks:
                continue

            # 2. Compute (One big MatMul)
            expert_input = torch.cat(batch_chunks, dim=0)
            expert_output = self.experts[local_expert_idx](expert_input)

            # 3. Scatter results back to their Source Ranks
            active_sizes = [s for s in chunk_sizes if s > 0]
            if active_sizes:
                splitted_out = expert_output.split(active_sizes)
                split_idx = 0
                for rank, size in enumerate(chunk_sizes):
                    if size > 0:
                        results_by_rank[rank].append(splitted_out[split_idx])
                        split_idx += 1

        # 4. Concatenate by Rank
        final_chunks = []
        for rank_list in results_by_rank:
            if rank_list:
                final_chunks.append(torch.cat(rank_list, dim=0))

        if not final_chunks:
            return torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)

        return torch.cat(final_chunks, dim=0)

    def combine_exchange_async(self, expert_output, recv_counts_matrix, send_splits):
        """
        Stage 4: Reverse Async Communication (Combine)
        Returns: (final_data, handle)
        """
        if self.world_size == 1:
            return expert_output, None

        # "Output" for combine is what we received in Dispatch
        output_splits = recv_counts_matrix.sum(dim=1).tolist()

        # "Input" for combine is what we sent in Dispatch
        input_splits = send_splits

        total_back = sum(input_splits)
        final_data = torch.empty(
            total_back, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )

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
        """
        Standard Forward Pass (Baseline).
        Blocks on async handles to behave synchronously.
        """
        # 1. Gate
        permuted, expert_counts, rev_idx, weights, shape = self.gate_and_sort(x)

        # 2. Dispatch
        recv_data, handle, (recv_counts, send_splits) = self.dispatch_exchange_async(
            permuted, expert_counts
        )
        if handle:
            handle.wait()  # Block if running standard benchmark

        # 3. Compute
        expert_out = self.compute_experts(recv_data, recv_counts)

        # 4. Combine
        final_data, handle = self.combine_exchange_async(expert_out, recv_counts, send_splits)
        if handle:
            handle.wait()  # Block if running standard benchmark

        # 5. Unpermute
        return self.unpermute(final_data, rev_idx, weights, shape)
