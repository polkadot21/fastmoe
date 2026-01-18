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

        assert num_experts % self.world_size == 0
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
        Returns:
          permuted_data: [B*T*k, D] sorted by global expert id (thus also grouped by rank)
          send_rank_counts: [world_size] tokens to send to each rank
          send_expert_counts: [world_size, num_local_experts] counts per (rank, local_expert)
          reverse_map_indices: [B*T*k] mapping back to original token rows (for unpermute)
          sorted_weights: [B*T*k] gate weights aligned to permuted_data
          (B, T): original shape
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [N, D], N=B*T

        # Router logits -> top-k
        logits = self.router(x_flat)  # [N, E]
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)  # [N,k]
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(x)

        # Flatten k dimension
        indices_flat = topk_indices.reshape(-1)  # [N*k]
        weights_flat = topk_weights.reshape(-1)  # [N*k]

        # Sanity: indices in range
        assert indices_flat.numel() == x_flat.shape[0] * self.top_k
        if torch._C._get_tracing_state() is None:  # avoid compile/tracing weirdness
            assert int(indices_flat.min().item()) >= 0
            assert int(indices_flat.max().item()) < self.num_experts

        # Sort by global expert id for contiguity
        sort_indices = torch.argsort(indices_flat, stable=True)
        expert_indices_sorted = indices_flat.index_select(0, sort_indices)  # [N*k]

        # Reverse map: which original token row each expanded item came from
        src_indices = torch.arange(
            x_flat.shape[0], device=x.device, dtype=torch.int64
        ).repeat_interleave(self.top_k)
        reverse_map_indices = src_indices.index_select(0, sort_indices).to(torch.int64)  # [N*k]

        sorted_weights = weights_flat.index_select(0, sort_indices).contiguous()
        permuted_data = x_flat.index_select(0, reverse_map_indices).contiguous()  # [N*k, D]

        # Counts per global expert (length = num_experts)
        global_expert_counts = torch.bincount(
            expert_indices_sorted.to(torch.int64), minlength=self.num_experts
        ).to(torch.int32)

        # Rank id for each sorted item: expert_id // num_local_experts
        rank_ids = torch.div(
            expert_indices_sorted.to(torch.int64), self.num_local_experts, rounding_mode="floor"
        )  # [N*k] int64 in [0, world_size-1]

        # send_rank_counts MUST be [world_size]
        send_rank_counts = (
            torch.bincount(rank_ids, minlength=self.world_size).to(torch.int32).contiguous()
        )

        # Per-rank per-local-expert counts: [world_size, num_local_experts]
        send_expert_counts = global_expert_counts.view(
            self.world_size, self.num_local_experts
        ).contiguous()

        # Fail fast if anything is off
        assert (
            send_rank_counts.numel() == self.world_size
        ), f"send_rank_counts wrong shape: {tuple(send_rank_counts.shape)}"
        assert send_expert_counts.shape == (
            self.world_size,
            self.num_local_experts,
        ), f"send_expert_counts wrong shape: {tuple(send_expert_counts.shape)}"
        # Also check total accounting matches actual send items
        assert (
            int(send_rank_counts.sum().item()) == permuted_data.shape[0]
        ), f"rank_counts sum {int(send_rank_counts.sum().item())} != permuted_data {permuted_data.shape[0]}"  # noqa

        return (
            permuted_data,
            send_rank_counts,  # FIXED: [world_size]
            send_expert_counts.to(torch.long),
            reverse_map_indices,  # [N*k]
            sorted_weights,  # [N*k]
            (B, T),
        )

    def dispatch_exchange(self, permuted_data, send_rank_counts, send_expert_counts):
        # Metadata: enforce int32 + contiguous + expected shapes
        send_rank_counts = send_rank_counts.to(torch.int32).contiguous().view(-1)
        assert (
            send_rank_counts.numel() == self.world_size
        ), f"send_rank_counts wrong shape: {tuple(send_rank_counts.shape)}"

        send_expert_counts = send_expert_counts.to(torch.int32).contiguous()
        assert send_expert_counts.shape == (
            self.world_size,
            self.num_local_experts,
        ), f"send_expert_counts wrong shape: {tuple(send_expert_counts.shape)}"

        # 1) exchange expert-count matrix
        recv_expert_counts = torch.empty_like(send_expert_counts)
        dist.all_to_all_single(recv_expert_counts, send_expert_counts, group=self.group)
        tokens_per_local_expert = recv_expert_counts.sum(dim=0).to(torch.int64).tolist()

        # 2) exchange rank counts vector
        recv_rank_counts = torch.empty_like(send_rank_counts)
        dist.all_to_all_single(recv_rank_counts, send_rank_counts, group=self.group)

        send_list = send_rank_counts.cpu().to(torch.int64).tolist()
        recv_list = recv_rank_counts.cpu().to(torch.int64).tolist()
        assert len(send_list) == self.world_size
        assert len(recv_list) == self.world_size

        total_recv = sum(recv_list)
        recv_data = torch.empty(
            total_recv, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )

        # 3) data all-to-all
        dist.all_to_all_single(
            recv_data,
            permuted_data,
            output_split_sizes=recv_list,
            input_split_sizes=send_list,
            group=self.group,
        )
        return recv_data, recv_list, send_list, tokens_per_local_expert

    def dispatch_exchange_static(self, permuted_data, static_splits, real_tokens_per_rank: int):
        if self.world_size == 1:
            return permuted_data, static_splits, static_splits

        total_send = sum(static_splits)

        if total_send > permuted_data.shape[0]:
            send_data = torch.zeros(
                total_send, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
            )
            padded_per_rank = static_splits[0]
            send_view = send_data.view(self.world_size, padded_per_rank, self.dim)
            src_view = permuted_data.view(self.world_size, real_tokens_per_rank, self.dim)
            send_view[:, :real_tokens_per_rank, :] = src_view
            send_data = send_data.view(-1, self.dim)
        else:
            send_data = permuted_data

        recv_data = torch.empty(
            total_send, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )
        dist.all_to_all_single(
            recv_data,
            send_data,
            output_split_sizes=static_splits,
            input_split_sizes=static_splits,
            group=self.group,
        )

        if total_send > permuted_data.shape[0]:
            padded_per_rank = static_splits[0]
            recv_view = recv_data.view(self.world_size, padded_per_rank, self.dim)
            real_recv = recv_view[:, :real_tokens_per_rank, :].reshape(-1, self.dim)
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
        if self.world_size == 1:
            return expert_output
        total_back = sum(send_splits)
        final_data = torch.empty(
            total_back, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )
        dist.all_to_all_single(
            final_data,
            expert_output,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.group,
        )
        return final_data

    def combine_exchange_static(self, expert_output, static_splits, real_tokens_per_rank: int):
        if self.world_size == 1:
            return expert_output

        total_send = sum(static_splits)
        if total_send > expert_output.shape[0]:
            send_data = torch.zeros(
                total_send, self.dim, device=expert_output.device, dtype=expert_output.dtype
            )
            send_data[: expert_output.shape[0]] = expert_output
        else:
            send_data = expert_output

        final_data_padded = torch.empty(
            total_send, self.dim, device=expert_output.device, dtype=expert_output.dtype
        )
        dist.all_to_all_single(
            final_data_padded,
            send_data,
            output_split_sizes=static_splits,
            input_split_sizes=static_splits,
            group=self.group,
        )

        if total_send > expert_output.shape[0]:
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
        permuted, send_rank_counts, send_exp_counts, rev_idx, weights, shape = self.gate_and_sort(x)
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
        stream0: torch.cuda.Stream,  # Gate/Attn compute (HIGH prio recommended)
        stream1: torch.cuda.Stream,  # Comm (LOW/NORMAL prio recommended)
        stream2: torch.cuda.Stream,  # Expert stream
        comm_balance_factor: int = 1,
        num_microbatches: int = 2,
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

        self.gate_stream = stream0
        self.comm_stream = stream1
        # Separate stream so gate/attn can keep running even if experts wait on dispatch
        self.expert_stream = stream2

        self.comm_balance_factor = comm_balance_factor
        self.num_microbatches = num_microbatches

        self.static_splits = None
        self.real_tokens_per_rank = 0

    def _init_static_splits(self, x_mb):
        if self.static_splits is not None:
            return

        B, T, _ = x_mb.shape
        total_tokens = B * T * self.moe.top_k

        if not dist.is_initialized():
            # Single-rank fallback
            self.real_tokens_per_rank = total_tokens
            self.static_splits = [total_tokens]
            return

        world_size = dist.get_world_size(self.moe.group)
        if world_size == 1:
            self.real_tokens_per_rank = total_tokens
            self.static_splits = [total_tokens]
            return

        self.real_tokens_per_rank = total_tokens // world_size
        padded_tokens_per_rank = self.real_tokens_per_rank * self.comm_balance_factor
        self.static_splits = [padded_tokens_per_rank] * world_size

    def forward(self, x):
        mbs = x.chunk(self.num_microbatches, dim=0)
        self._init_static_splits(mbs[0])

        n = len(mbs)
        ctx = [{} for _ in range(n)]
        outs = [None] * n

        # Events per microbatch
        ev_gate_done = [torch.cuda.Event() for _ in range(n)]
        ev_dispatch_done = [torch.cuda.Event() for _ in range(n)]
        ev_expert_done = [torch.cuda.Event() for _ in range(n)]
        ev_combine_done = [torch.cuda.Event() for _ in range(n)]

        # ---- Pipeline schedule (steady-state) ----
        # For i:
        #   Gate(i)           on gate_stream
        #   Dispatch(i)       on comm_stream    (wait Gate(i))
        #   Experts(i-1)      on expert_stream  (wait Dispatch(i-1))
        #   Combine(i-1)      on comm_stream    (wait Experts(i-1))
        #
        # This preserves your “dispatch next before waiting on experts” intention,
        # but avoids blocking future gates behind expert waits.

        for i, mb in enumerate(mbs):
            # 1) Gate/Attn i
            with torch.cuda.stream(self.gate_stream):
                with record_function(f"Comp: Attn/Gate {i}"):
                    h = self.norm1(mb)
                    h = self.attn(h)
                    mb_resid = mb + h
                    moe_in = self.norm2(mb_resid)
                    perm, _, _, rev, w, s = self.moe.gate_and_sort(moe_in)
                    ctx[i].update({"perm": perm, "rev": rev, "w": w, "s": s, "mb_resid": mb_resid})
                ev_gate_done[i].record()

            # 2) Dispatch i (queued early; comm stream waits for gate event)
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ev_gate_done[i])
                with record_function(f"Comm: Dispatch {i}"):
                    rd, _, _ = self.moe.dispatch_exchange_static(
                        ctx[i]["perm"], self.static_splits, self.real_tokens_per_rank
                    )
                    # free send buffer early
                    del ctx[i]["perm"]
                    ctx[i]["rd"] = rd
                ev_dispatch_done[i].record()

            # 3) Experts (i-1)
            j = i - 1
            if j >= 0:
                with torch.cuda.stream(self.expert_stream):
                    self.expert_stream.wait_event(ev_dispatch_done[j])
                    with record_function(f"Comp: Experts {j}"):
                        eo = self.moe.compute_experts_static(ctx[j]["rd"])
                        del ctx[j]["rd"]
                        ctx[j]["eo"] = eo
                    ev_expert_done[j].record()

                # 4) Combine (i-1) AFTER we’ve already enqueued Dispatch(i)
                with torch.cuda.stream(self.comm_stream):
                    self.comm_stream.wait_event(ev_expert_done[j])
                    with record_function(f"Comm: Combine {j}"):
                        fd = self.moe.combine_exchange_static(
                            ctx[j]["eo"], self.static_splits, self.real_tokens_per_rank
                        )
                        del ctx[j]["eo"]
                        ctx[j]["fd"] = fd
                    ev_combine_done[j].record()

        # Flush last microbatch experts+combine
        last = n - 1
        with torch.cuda.stream(self.expert_stream):
            self.expert_stream.wait_event(ev_dispatch_done[last])
            with record_function(f"Comp: Experts {last}"):
                eo = self.moe.compute_experts_static(ctx[last]["rd"])
                del ctx[last]["rd"]
                ctx[last]["eo"] = eo
            ev_expert_done[last].record()

        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ev_expert_done[last])
            with record_function(f"Comm: Combine {last}"):
                fd = self.moe.combine_exchange_static(
                    ctx[last]["eo"], self.static_splits, self.real_tokens_per_rank
                )
                del ctx[last]["eo"]
                ctx[last]["fd"] = fd
            ev_combine_done[last].record()

        # Finalize (unpermute + residual add)
        with torch.cuda.stream(self.gate_stream):
            for i in range(n):
                self.gate_stream.wait_event(ev_combine_done[i])
                with record_function(f"Comp: Finalize {i}"):
                    res = self.moe.unpermute(ctx[i]["fd"], ctx[i]["rev"], ctx[i]["w"], ctx[i]["s"])
                    out = res + ctx[i]["mb_resid"]
                    outs[i] = out
                    # cleanup
                    del ctx[i]["fd"], ctx[i]["rev"], ctx[i]["w"], ctx[i]["s"], ctx[i]["mb_resid"]

        # Make sure result is ready on the current stream
        torch.cuda.current_stream().wait_stream(self.gate_stream)
        return torch.cat(outs, dim=0)


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
        stream2: torch.cuda.Stream | None,
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
                        stream2=stream2,
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
