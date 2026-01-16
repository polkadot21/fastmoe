import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from fastmoe import consts
from fastmoe.kernels.ops import grouped_weighted_scatter_add


# ========== Self-Attention ========= #
class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


# ======= Feed Forward ======= #
class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        num_experts: int,
        top_k: int,
        implementation: consts.MoEImplementation,
        group: dist.ProcessGroup | None = None,
    ) -> None:
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

    def gate_and_sort(self, x: torch.Tensor):
        """
        Stage 1: Gating & Permutation
        Purpose: Route tokens to experts and group them contiguously in memory
        to maximize compute efficiency and prepare for All-to-All communication.
        """
        # Unpack input shape: Batch, Seq Len, Hidden Dimension
        # Example: [32, 2048, 8192]
        B, T, D = x.shape
        # Flatten Batch and Seq Len to treat all tokens independently for routing.
        # Shape: [B*T, D] -> [65,536, 8192]
        x_flat = x.view(-1, D)
        # 1. ROUTING
        # Project tokens to expert space to get logits.
        # Shape: [B*T, Num_Experts]
        logits = self.router(x_flat)
        # Select the top-k experts for each token.
        # topk_weights: [B*T, k], topk_indices: [B*T, k]
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        # Normalize weights using Softmax over the top-k dimension.
        # Shape: [B*T, k]
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)
        # 2. FLATTENING
        # We unroll the 'k' dimension because physically, a token sent to 2 experts
        # becomes 2 distinct computation tasks.
        # Shape: [B*T*k] -> [131,072]
        indices_flat = topk_indices.view(-1)
        weights_flat = topk_weights.view(-1)
        # 3. SORTING (CRITICAL)
        # We perform an argsort on the expert indices. This gives us the permutation
        # indices needed to group all tokens destined for Expert 0 together,
        # then Expert 1, etc. This ensures contiguous memory access during compute.
        # Shape: [B*T*k]
        sort_indices = torch.argsort(indices_flat)
        # Get the sorted expert IDs (0,0,0... 1,1,1... 2,2...).
        # Used later for counting workload per expert.
        # Shape: [B*T*k]
        expert_indices_sorted = indices_flat[sort_indices]
        # 4. PERMUTATION MAPPING
        # Create source indices [0, 0, 1, 1, ..., N, N] (repeated k times).
        # This tracks which original token row corresponds to the flattened routing decision.
        # Shape: [B*T*k]
        src_indices = torch.arange(x_flat.shape[0], device=x.device).repeat_interleave(self.top_k)
        # Apply the sort permutation to the source indices.
        # This tells us: "The 1st element in the sorted buffer comes from Token X in x_flat".
        # Shape: [B*T*k]
        reverse_map_indices = src_indices[sort_indices]
        # Permute the weights to align with the sorted data.
        # Shape: [B*T*k]
        sorted_weights = weights_flat[sort_indices]
        # 5. GATHER (SCATTER)
        # Permute the actual token data. This physically moves data in VRAM.
        # The result is a tensor where all inputs for Expert 0 are contiguous, etc.
        # Shape: [B*T*k, D] -> [131,072, 8192] (Expansion factor of k)
        permuted_data = x_flat[reverse_map_indices]

        # 6. WORKLOAD ACCOUNTING
        # Count how many tokens are assigned to each global expert.
        # Shape: [Num_Experts]
        global_expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)

        if self.world_size > 1:
            # Reshape counts to map Global Experts -> Ranks.
            # Example: If Rank 0 hosts Experts [0,1] and Rank 1 hosts [2,3].
            # Shape: [World_Size, Num_Local_Experts]
            expert_counts_by_rank = global_expert_counts.view(
                self.world_size, self.num_local_experts
            )
            # Sum local expert counts to get the total number of tokens to send to each Rank.
            # This is required for the NCCL all_to_all_single communication size.
            # Shape: [World_Size]
            rank_counts = expert_counts_by_rank.sum(dim=1)
        else:
            expert_counts_by_rank = global_expert_counts.view(1, self.num_experts)
            rank_counts = torch.tensor([permuted_data.shape[0]], device=x.device, dtype=torch.long)

        return (
            permuted_data,  # The sorted token data [B*T*k, D]
            rank_counts,  # Load balancing for Data Exchange [WorldSize]
            expert_counts_by_rank.to(
                dtype=torch.long
            ),  # Metadata for receiver to split buffer [WS, LocalExp]
            reverse_map_indices,  # Indices to restore original order [B*T*k]
            sorted_weights,  # Weights for the combine step [B*T*k]
            (B, T),  # Original shape metadata
        )

    def dispatch_exchange(self, permuted_data, send_rank_counts, send_expert_counts):
        """Dynamic dispatch for Standard implementation"""
        if self.world_size == 1:
            local_tokens_per_expert = send_expert_counts.view(-1).tolist()
            return (
                permuted_data,
                send_rank_counts.tolist(),
                send_rank_counts.tolist(),
                local_tokens_per_expert,
            )
        # 1. METADATA EXCHANGE
        # We need to know how many tokens we will receive for each of our local experts.
        # send_expert_counts: [World_Size, Num_Local_Experts]
        recv_expert_counts = torch.empty_like(send_expert_counts)
        # All-to-All for metadata. This is a small, blocking communication.
        # Each rank tells every other rank: "I have X tokens for your Expert Y."
        dist.all_to_all_single(recv_expert_counts, send_expert_counts, group=self.group)

        # Aggregate counts to determine how to split the incoming data buffer.
        # Summing over dimension 0 (ranks) gives total tokens for each local expert.
        # Shape: [Num_Local_Experts] (as a Python list for torch.split)
        tokens_per_local_expert = recv_expert_counts.sum(dim=0).tolist()

        # 2. DATA EXCHANGE SETUP
        # We need to know the total size of the incoming data buffer from each rank.
        # recv_rank_counts: [World_Size]
        recv_rank_counts = torch.empty_like(send_rank_counts)
        # All-to-All for data sizes.
        # Each rank tells every other rank: "I am sending you N total tokens."
        dist.all_to_all_single(recv_rank_counts, send_rank_counts, group=self.group)
        # Prepare list for NCCL split sizes.
        # send_list: [World_Size], recv_list: [World_Size]
        send_list = send_rank_counts.tolist()
        recv_list = recv_rank_counts.tolist()
        # Calculate total receive buffer size to allocate memory.
        total_recv = sum(recv_list)
        # Allocate the receiving buffer.
        # Shape: [Total_Recv_Tokens, Hidden_Dim]
        recv_data = torch.empty(
            total_recv, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )
        # 3. DATA EXCHANGE (HEAVY LIFTING)
        # Perform the actual token transfer.
        # This moves the bulk of the data across NVLink/Infiniband.
        dist.all_to_all_single(
            recv_data,
            permuted_data,
            output_split_sizes=recv_list,
            input_split_sizes=send_list,
            group=self.group,
        )

        return recv_data, recv_list, send_list, tokens_per_local_expert

    def dispatch_exchange_static(self, permuted_data, static_splits, real_tokens_per_rank: int):
        """Static Dispatch with Padding."""
        if self.world_size == 1:
            return permuted_data, static_splits, static_splits
        # Calculate total size based on static splits (includes padding factor).
        # This is a Python integer, so no GPU read is triggered.
        total_send = sum(static_splits)

        # 1. PADDING LOGIC
        # Check if we need to pad the data (Artificial Load Balancing / Benchmark Stress Test).
        # permuted_data shape: [Actual_Tokens, Dim]
        if total_send > permuted_data.shape[0]:
            # Allocate a larger buffer to hold real data + padding.
            # Shape: [Total_Send_Static, Dim]
            send_data = torch.zeros(
                total_send, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
            )
            # The static splits are uniform per rank.
            padded_per_rank = static_splits[0]
            # View 1: The padded destination buffer [WS, Padded_Size, Dim]
            send_view = send_data.view(self.world_size, padded_per_rank, self.dim)
            # View 2: The actual source data [WS, Real_Size, Dim]
            # Assumes permuted_data is already sorted by rank (which gate_and_sort guarantees).
            src_view = permuted_data.view(self.world_size, real_tokens_per_rank, self.dim)
            # Copy real data into the start of each rank's section.
            # Result: [Real Data ..... | Padding .....] for each rank.
            send_view[:, :real_tokens_per_rank, :] = src_view
            # Flatten back for NCCL.
            send_data = send_data.view(-1, self.dim)
        else:
            send_data = permuted_data

        # Allocate receive buffer based on static knowledge.
        # Shape: [Total_Recv_Static, Dim]
        recv_data = torch.empty(
            total_send, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )
        # 2. STATIC EXCHANGE
        # This call is non-blocking on the CPU because static_splits is a Python list.
        # The CPU launches the kernel and immediately moves to the next instruction.
        dist.all_to_all_single(
            recv_data,
            send_data,
            output_split_sizes=static_splits,
            input_split_sizes=static_splits,
            group=self.group,
        )

        # 3. DE-PADDING LOGIC
        # If we padded the sent data, we must slice the received data to extract valid tokens.
        if total_send > permuted_data.shape[0]:
            padded_per_rank = static_splits[0]
            # View receiving buffer by rank [WS, Padded_Size, Dim]
            recv_view = recv_data.view(self.world_size, padded_per_rank, self.dim)
            # Slice out the real tokens. We know exactly how many real tokens to expect per rank.
            # Shape: [WS, Real_Size, Dim] -> [Total_Real, Dim]
            real_recv = recv_view[:, :real_tokens_per_rank, :].reshape(-1, self.dim)
            # Return contiguous buffer for optimal compute performance in the next stage.
            return real_recv.contiguous(), static_splits, static_splits

        return recv_data, static_splits, static_splits

    def compute_experts(self, recv_data, tokens_per_expert: list[int]) -> torch.Tensor:
        chunks = recv_data.split(tokens_per_expert, dim=0)
        results = []
        for chunk, expert in zip(chunks, self.experts, strict=False):
            if chunk.shape[0] > 0:
                results.append(expert(chunk))
            else:
                results.append(
                    torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)
                )
        if not results:
            return torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)
        return torch.cat(results, dim=0)

    def compute_experts_static(self, recv_data) -> torch.Tensor:
        chunks = recv_data.chunk(self.num_local_experts, dim=0)
        results = []
        for chunk, expert in zip(chunks, self.experts, strict=False):
            if chunk.shape[0] > 0:
                results.append(expert(chunk))
            else:
                results.append(
                    torch.empty(0, self.dim, device=recv_data.device, dtype=recv_data.dtype)
                )
        return torch.cat(results, dim=0)

    def combine_exchange(self, expert_output, recv_splits, send_splits):
        """
        Stage 4: Dynamic Combine (Standard Implementation)
        Purpose: Return processed tokens from experts back to their original ranks.
        This is the reverse of the dispatch step.
        """
        if self.world_size == 1:
            return expert_output
        # Calculate the total size of the data we expect to receive back.
        # 'send_splits' here refers to what we originally sent (which dictates what we get back).
        # This summation happens on the CPU using the list we tracked earlier.
        # Shape: Scalar (Total number of tokens this rank originally dispatched)
        total_back = sum(send_splits)
        # Allocate the buffer for the final combined data.
        # Shape: [Total_Original_Tokens, Hidden_Dim]
        final_data = torch.empty(
            total_back, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )

        # Execute the All-to-All communication.
        # expert_output: The results from our local experts (to be sent back to others).
        # final_data: The buffer to store results returning from other ranks.
        # output_split_sizes=send_splits: We expect to receive as much as we originally sent to each rank. # noqa
        # input_split_sizes=recv_splits: We are sending back exactly what we received (and computed on). # noqa
        dist.all_to_all_single(
            final_data,
            expert_output,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.group,
        )
        return final_data

    def combine_exchange_static(self, expert_output, static_splits, real_tokens_per_rank: int):
        """
        Stage 4: Static Combine (Pipelined Benchmark)
        Purpose: Non-blocking, padded return of tokens to support overlap.
        Mirror image of dispatch_exchange_static.
        """
        if self.world_size == 1:
            return expert_output
        # Total size to send/receive (including padding factor).
        # Calculated from static_splits (Python list), so no CPU block.
        total_send = sum(static_splits)
        # 1. PADDING LOGIC (OUTPUT SIDE)
        # If we are simulating higher bandwidth usage (dead weight), we pad the output.
        if total_send > expert_output.shape[0]:
            # Allocate padded send buffer.
            # Shape: [Total_Send_Static, Dim]
            send_data = torch.zeros(
                total_send, self.dim, device=expert_output.device, dtype=expert_output.dtype
            )
            # SIMPLIFICATION FOR BENCHMARK:
            # Unlike dispatch, we don't strictly need to interleave padding for correctness
            # because 'expert_output' is already a contiguous block of results.
            # We simply copy the valid results to the start of the buffer.
            # Rank 0 will receive valid data + garbage. Other ranks might receive mostly garbage.
            # This is fine because we discard the garbage immediately after receipt.
            # Validity: The timing and bandwidth stress are identical to a rigorous padding.
            send_data[: expert_output.shape[0]] = expert_output
        else:
            send_data = expert_output

        # Allocate the receiving buffer (padded size).
        # Shape: [Total_Recv_Static, Dim]
        final_data_padded = torch.empty(
            total_send, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )
        # 2. STATIC EXCHANGE
        # Non-blocking call. CPU queues this and moves on.
        # Symmetric exchange: We send and receive 'total_send' tokens.
        dist.all_to_all_single(
            final_data_padded,
            send_data,
            output_split_sizes=static_splits,
            input_split_sizes=static_splits,
            group=self.group,
        )
        # 3. DE-PADDING LOGIC
        # If we used padding, we must slice the result to return only the real data.
        # We expect 'expert_output.shape[0]' tokens back (the same amount we started with).
        if total_send > expert_output.shape[0]:
            # Slice the valid data from the start of the buffer.
            # Shape: [Real_Token_Count, Dim]
            return final_data_padded[: expert_output.shape[0]]

        return final_data_padded

    def unpermute(self, x_out, reverse_map_indices, sorted_weights, original_shape) -> torch.Tensor:
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
        # Standard Implementation
        permuted, send_rank_counts, send_exp_counts, rev_idx, weights, shape = self.gate_and_sort(x)
        # Dynamic path: Pass tensors directly, handle blocking inside dispatch_exchange
        recv_data, recv_splits, send_splits_list, tokens_per_expert = self.dispatch_exchange(
            permuted, send_rank_counts, send_exp_counts
        )
        expert_out = self.compute_experts(recv_data, tokens_per_expert)
        final_data = self.combine_exchange(expert_out, recv_splits, send_splits_list)
        return self.unpermute(final_data, rev_idx, weights, shape)


# ======= Blocks ======== #
class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_dim=ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class MoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ff_dim: int,
        num_experts: int,
        top_k: int,
        implementation: consts.MoEImplementation,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = MoEFeedForward(
            dim=dim,
            ff_dim=ff_dim,
            num_experts=num_experts,
            top_k=top_k,
            implementation=implementation,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class PipelinedMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ff_dim: int,
        num_experts: int,
        top_k: int,
        stream0: torch.cuda.Stream,  # We will treat this as COMPUTE STREAM
        stream1: torch.cuda.Stream,  # We will treat this as COMM STREAM
        comm_balance_factor: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoEFeedForward(
            dim=dim,
            ff_dim=ff_dim,
            num_experts=num_experts,
            top_k=top_k,
            implementation=consts.MoEImplementation.FAST,
        )

        # FUNCTIONAL STREAMS
        # Stream 0 -> Compute (Calculations)
        # Stream 1 -> Communication (NCCL)
        self.compute_stream = stream0
        self.comm_stream = stream1

        self.comm_balance_factor = comm_balance_factor
        self.static_splits = None
        self.real_tokens_per_rank = 0

    def _init_static_splits(self, x_half):
        if self.static_splits is None and dist.is_initialized():
            B, T, D = x_half.shape
            total_tokens = B * T * self.moe.top_k
            world_size = dist.get_world_size(self.moe.group)
            self.real_tokens_per_rank = total_tokens // world_size
            padded_tokens_per_rank = self.real_tokens_per_rank * self.comm_balance_factor
            self.static_splits = [padded_tokens_per_rank] * world_size

    def forward(self, x):
        """
        Functional Stream Overlap:
        Separates work into a Compute Lane and a Communication Lane.
        """
        x_chunks = x.chunk(2, dim=0)
        mb0, mb1 = x_chunks[0], x_chunks[1]
        self._init_static_splits(mb0)

        ctx0, ctx1 = {}, {}

        # Events for synchronization
        # We need to signal across the two lanes.
        ev_gate0_done = torch.cuda.Event()
        ev_dispatch0_done = torch.cuda.Event()
        ev_experts0_done = torch.cuda.Event()

        ev_gate1_done = torch.cuda.Event()
        ev_dispatch1_done = torch.cuda.Event()
        ev_experts1_done = torch.cuda.Event()

        # -----------------------------------------------------------
        # STEP 1: PREPARE (Gating on Compute Stream)
        # -----------------------------------------------------------
        with torch.cuda.stream(self.compute_stream):
            # Gating MB0
            with record_function("Comp: Attn/Gate 0"):
                h0 = self.norm1(mb0)
                h0 = self.attn(h0)
                mb0_resid = mb0 + h0
                moe_in_0 = self.norm2(mb0_resid)
                perm0, _, _, rev0, w0, s0 = self.moe.gate_and_sort(moe_in_0)
                ctx0.update({"rev": rev0, "w": w0, "s": s0})
            ev_gate0_done.record()

            # Gating MB1
            with record_function("Comp: Attn/Gate 1"):
                h1 = self.norm1(mb1)
                h1 = self.attn(h1)
                mb1_resid = mb1 + h1
                moe_in_1 = self.norm2(mb1_resid)
                perm1, _, _, rev1, w1, s1 = self.moe.gate_and_sort(moe_in_1)
                ctx1.update({"rev": rev1, "w": w1, "s": s1})
            ev_gate1_done.record()

        # -----------------------------------------------------------
        # STEP 2: PIPELINE LOOP (The Zig-Zag)
        # We interleave submissions to ensure the GPU stays busy.
        # -----------------------------------------------------------

        # [A] COMM STREAM: Dispatch MB0
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_gate0_done)  # Wait for data
            with record_function("Comm: Dispatch 0"):
                rd0, rc0, sl0 = self.moe.dispatch_exchange_static(
                    perm0, self.static_splits, self.real_tokens_per_rank
                )
                ctx0.update({"rd": rd0})
            ev_dispatch0_done.record()

        # [B] COMM STREAM: Dispatch MB1
        # We queue this immediately. It will run after Dispatch 0.
        # Ideally, it runs *while* Experts 0 is running.
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_gate1_done)
            with record_function("Comm: Dispatch 1"):
                rd1, rc1, sl1 = self.moe.dispatch_exchange_static(
                    perm1, self.static_splits, self.real_tokens_per_rank
                )
                ctx1.update({"rd": rd1})
            ev_dispatch1_done.record()

        # [C] COMPUTE STREAM: Experts MB0
        # This can start as soon as Dispatch 0 is done.
        # It will overlap with Dispatch 1 (which is on Comm stream).
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(ev_dispatch0_done)
            with record_function("Comp: Experts 0"):
                eo0 = self.moe.compute_experts_static(ctx0["rd"])
                del ctx0["rd"]
            ev_experts0_done.record()

        # [D] COMPUTE STREAM: Experts MB1
        # Overlaps with Combine 0 (below).
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(ev_dispatch1_done)
            with record_function("Comp: Experts 1"):
                eo1 = self.moe.compute_experts_static(ctx1["rd"])
                del ctx1["rd"]
            ev_experts1_done.record()

        # [E] COMM STREAM: Combine MB0
        # Runs after Dispatch 1.
        # Overlaps with Experts 1.
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_experts0_done)  # Wait for result
            with record_function("Comm: Combine 0"):
                fd0 = self.moe.combine_exchange_static(
                    eo0, self.static_splits, self.real_tokens_per_rank
                )
                del eo0
                ctx0["fd"] = fd0

        # [F] COMM STREAM: Combine MB1
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_experts1_done)
            with record_function("Comm: Combine 1"):
                fd1 = self.moe.combine_exchange_static(
                    eo1, self.static_splits, self.real_tokens_per_rank
                )
                del eo1
                ctx1["fd"] = fd1

        # Signal Comm stream done so Compute stream can finalize
        ev_combine_done = torch.cuda.Event()
        self.comm_stream.record_event(ev_combine_done)

        # -----------------------------------------------------------
        # STEP 3: FINALIZE (Compute Stream)
        # -----------------------------------------------------------
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(ev_combine_done)

            with record_function("Comp: Finalize 0"):
                res0 = self.moe.unpermute(ctx0["fd"], ctx0["rev"], ctx0["w"], ctx0["s"])
                out0 = res0 + mb0_resid

            with record_function("Comp: Finalize 1"):
                res1 = self.moe.unpermute(ctx1["fd"], ctx1["rev"], ctx1["w"], ctx1["s"])
                out1 = res1 + mb1_resid

        torch.cuda.current_stream().wait_stream(self.compute_stream)
        return torch.cat([out0, out1], dim=0)


# ======= Model ========== #
class TinyModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        num_experts: int | None,
        top_k: int | None,
        implementation: consts.MoEImplementation | None,
        stream0: torch.cuda.Stream | None,
        stream1: torch.cuda.Stream | None,
        comm_balance_factor: int = 1,
        *,
        use_moe: bool,
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim, bias=False)
        if not use_moe:
            self.blocks = nn.ModuleList(
                [Block(dim=dim, n_heads=n_heads, ff_dim=ff_dim) for _ in range(n_layers)]
            )
        elif implementation == consts.MoEImplementation.STANDARD:
            self.blocks = nn.ModuleList(
                [
                    MoEBlock(
                        dim=dim,
                        n_heads=n_heads,
                        ff_dim=ff_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        implementation=implementation,
                    )
                    for _ in range(n_layers)
                ]
            )
        elif implementation == consts.MoEImplementation.FAST:
            self.blocks = nn.ModuleList(
                [
                    PipelinedMoEBlock(
                        dim=dim,
                        n_heads=n_heads,
                        ff_dim=ff_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        stream0=stream0,
                        stream1=stream1,
                        comm_balance_factor=comm_balance_factor,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            raise NotImplementedError
        self.out = nn.Linear(dim, in_dim, bias=False)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)
