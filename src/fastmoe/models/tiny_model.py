import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from fastmoe import consts
from fastmoe.kernels.ops import grouped_weighted_scatter_add


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
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

        global_expert_counts = torch.bincount(expert_indices_sorted, minlength=self.num_experts)

        if self.world_size > 1:
            expert_counts_by_rank = global_expert_counts.view(
                self.world_size, self.num_local_experts
            )
            rank_counts = expert_counts_by_rank.sum(dim=1)
        else:
            expert_counts_by_rank = global_expert_counts.view(1, self.num_experts)
            rank_counts = torch.tensor([permuted_data.shape[0]], device=x.device, dtype=torch.long)

        return (
            permuted_data,
            rank_counts,
            expert_counts_by_rank.to(dtype=torch.long),
            reverse_map_indices,
            sorted_weights,
            (B, T),
        )

    def dispatch_exchange(self, permuted_data, send_rank_counts_list, send_expert_counts_tensor):
        """
        Accepts PRE-CONVERTED lists for rank counts to avoid blocking inside.
        send_expert_counts_tensor is still a tensor because we need it
        for the async metadata exchange.
        """
        if self.world_size == 1:
            local_tokens_per_expert = send_expert_counts_tensor.view(-1).tolist()
            return (
                permuted_data,
                send_rank_counts_list,
                send_rank_counts_list,
                local_tokens_per_expert,
            )

        # 1. Metadata Exchange (Async)
        recv_expert_counts = torch.empty_like(send_expert_counts_tensor)
        meta_work = dist.all_to_all_single(
            recv_expert_counts, send_expert_counts_tensor, group=self.group, async_op=True
        )

        meta_work.wait()
        tokens_per_local_expert = recv_expert_counts.sum(dim=0).tolist()

        # 2. Data Exchange (Using pre-computed lists)
        # We don't need to read send_rank_counts_list from GPU here, it's passed in.
        total_recv = sum(
            send_rank_counts_list
        )  # Assuming balanced exchange for allocation size estimate or...
        # Wait, for All-to-All we need to know how much WE receive.
        # We need an All-to-All for counts.

        # We must perform the count exchange.
        # Ideally, we would have done this asynchronously too.
        # But `dist.all_to_all_single` for counts is fast.

        # Convert list back to tensor for communication if needed,
        # OR just use the tensor if we had it.
        # Actually, let's keep it simple: We communicate the counts.
        # But we need `recv_rank_counts` (LIST) to allocate output.

        # To avoid re-blocking, we assume the user passed valid lists.
        # But we need Recv Counts.

        send_rank_counts_tensor = torch.tensor(
            send_rank_counts_list, device=permuted_data.device, dtype=torch.long
        )
        recv_rank_counts_tensor = torch.empty_like(send_rank_counts_tensor)
        dist.all_to_all_single(recv_rank_counts_tensor, send_rank_counts_tensor, group=self.group)

        recv_list = recv_rank_counts_tensor.tolist()
        total_recv = sum(recv_list)

        recv_data = torch.empty(
            total_recv, self.dim, device=permuted_data.device, dtype=permuted_data.dtype
        )

        dist.all_to_all_single(
            recv_data,
            permuted_data,
            output_split_sizes=recv_list,
            input_split_sizes=send_rank_counts_list,
            group=self.group,
        )

        return recv_data, recv_list, send_rank_counts_list, tokens_per_local_expert

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

    def combine_exchange(self, expert_output, recv_splits, send_splits):
        if self.world_size == 1:
            return expert_output

        if isinstance(send_splits, torch.Tensor):
            send_splits = send_splits.tolist()

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
        # Convert to list here for standard path
        s_list = send_rank_counts.tolist()
        recv_data, recv_splits, send_splits_list, tokens_per_expert = self.dispatch_exchange(
            permuted, s_list, send_exp_counts
        )
        expert_out = self.compute_experts(recv_data, tokens_per_expert)
        final_data = self.combine_exchange(expert_out, recv_splits, send_splits_list)
        return self.unpermute(final_data, rev_idx, weights, shape)


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
        stream0: torch.cuda.Stream,
        stream1: torch.cuda.Stream,
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
        self.stream0 = stream0
        self.stream1 = stream1

    def forward(self, x):
        x_chunks = x.chunk(2, dim=0)
        mb0, mb1 = x_chunks[0], x_chunks[1]

        ev_attn0_done = torch.cuda.Event()
        ctx0, ctx1 = {}, {}

        # --- 1. S0: Attn & Gate ---
        with torch.cuda.stream(self.stream0):
            with record_function("S0: Attn 0"):
                h0 = self.norm1(mb0)
                h0 = self.attn(h0)
                mb0_resid = mb0 + h0
                moe_in_0 = self.norm2(mb0_resid)
                perm0, sc0, exp_c0, rev0, w0, s0 = self.moe.gate_and_sort(moe_in_0)
                ctx0.update({"rev": rev0, "w": w0, "s": s0})

                # ASYNC D2H
                sc0_cpu = sc0.to("cpu", non_blocking=True)

            ev_attn0_done.record()

        # --- 2. S1: Attn & Gate (PRE-QUEUE) ---
        with torch.cuda.stream(self.stream1):
            self.stream1.wait_event(ev_attn0_done)
            with record_function("S1: Attn 1"):
                h1 = self.norm1(mb1)
                h1 = self.attn(h1)
                mb1_resid = mb1 + h1
                moe_in_1 = self.norm2(mb1_resid)
                perm1, sc1, exp_c1, rev1, w1, s1 = self.moe.gate_and_sort(moe_in_1)
                ctx1.update({"rev": rev1, "w": w1, "s": s1})

                # ASYNC D2H
                sc1_cpu = sc1.to("cpu", non_blocking=True)

        # --- 3. S0: Dispatch (CPU BLOCK 1) ---
        # This will block until sc0_cpu is ready.
        # But S1 Attn is already queued!
        s0_list = sc0_cpu.tolist()

        with torch.cuda.stream(self.stream0):
            with record_function("S0: Dispatch 0"):
                rd0, rc0, sl0, tpe0 = self.moe.dispatch_exchange(perm0, s0_list, exp_c0)
                ctx0.update({"rd": rd0, "rc": rc0, "sl": sl0, "tpe": tpe0})

        # --- 4. S0: Experts ---
        with torch.cuda.stream(self.stream0):
            with record_function("S0: Experts 0"):
                eo0 = self.moe.compute_experts(ctx0["rd"], ctx0["tpe"])
                del ctx0["rd"]

        # --- 5. S1: Dispatch (CPU BLOCK 2) ---
        s1_list = sc1_cpu.tolist()

        with torch.cuda.stream(self.stream1):
            with record_function("S1: Dispatch 1"):
                rd1, rc1, sl1, tpe1 = self.moe.dispatch_exchange(perm1, s1_list, exp_c1)
                ctx1.update({"rd": rd1, "rc": rc1, "sl": sl1, "tpe": tpe1})

        # --- 6. S1: Experts ---
        with torch.cuda.stream(self.stream1):
            with record_function("S1: Experts 1"):
                eo1 = self.moe.compute_experts(ctx1["rd"], ctx1["tpe"])
                del ctx1["rd"]

        # --- 7. S0: Combine ---
        with torch.cuda.stream(self.stream0):
            with record_function("S0: Combine 0"):
                fd0 = self.moe.combine_exchange(eo0, ctx0["rc"], ctx0["sl"])
                del eo0
            with record_function("S0: Finalize 0"):
                res0 = self.moe.unpermute(fd0, ctx0["rev"], ctx0["w"], ctx0["s"])
                out0 = res0 + mb0_resid
                del ctx0

        # --- 8. S1: Combine ---
        with torch.cuda.stream(self.stream1):
            with record_function("S1: Combine 1"):
                fd1 = self.moe.combine_exchange(eo1, ctx1["rc"], ctx1["sl"])
                del eo1
            with record_function("S1: Finalize 1"):
                res1 = self.moe.unpermute(fd1, ctx1["rev"], ctx1["w"], ctx1["s"])
                out1 = res1 + mb1_resid
                del ctx1

        torch.cuda.current_stream().wait_stream(self.stream0)
        torch.cuda.current_stream().wait_stream(self.stream1)
        return torch.cat([out0, out1], dim=0)


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
