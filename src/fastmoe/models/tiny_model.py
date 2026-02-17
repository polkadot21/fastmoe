import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import Config


# ==========================================
# Modules (SelfAttention, Expert) - Unchanged
# ==========================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        x = self.norm(x)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        return out + residual


class Expert(nn.Module):
    def __init__(self, dim: int, proj_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, proj_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# MoE Overlap Function (Orchestrator)
# ==========================================
class MoEOverlapFunction(Function):
    @staticmethod
    def forward(ctx, x, block):
        ctx.block = block

        # Split into micro-batches
        chunks = x.chunk(block.n_mb, dim=0)
        MB = block.n_mb

        # Contexts for each MB to store intermediate tensors
        fwd_ctx = [{} for _ in range(MB)]
        outputs = [None] * MB

        # CUDA Events for Synchronization
        ev_pre = [torch.cuda.Event() for _ in range(MB)]
        ev_disp = [torch.cuda.Event() for _ in range(MB)]
        ev_exp = [torch.cuda.Event() for _ in range(MB)]
        ev_comb = [torch.cuda.Event() for _ in range(MB)]

        # Tick Loop (Pipeline Schedule)
        total_ticks = MB + 4
        for tick in range(total_ticks):
            # Compute Stages
            block._fwd_stage_post_ops(tick - 4, fwd_ctx, outputs, ev_comb, chunks)
            block._fwd_stage_experts(tick - 2, fwd_ctx, ev_disp, ev_exp)
            block._fwd_stage_pre_ops(tick, fwd_ctx, chunks, ev_pre)

            # Comm Stages
            block._fwd_stage_combine(tick - 3, fwd_ctx, ev_exp, ev_comb)
            block._fwd_stage_dispatch(tick - 1, fwd_ctx, ev_pre, ev_disp)

        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMM])

        ctx.fwd_ctx = fwd_ctx
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        block = ctx.block
        fwd_ctx = ctx.fwd_ctx
        MB = block.n_mb

        grad_chunks = grad_output.chunk(MB, dim=0)

        # Backward Events
        ev_post = [torch.cuda.Event() for _ in range(MB)]
        ev_comb = [torch.cuda.Event() for _ in range(MB)]
        ev_exp = [torch.cuda.Event() for _ in range(MB)]
        ev_disp = [torch.cuda.Event() for _ in range(MB)]

        dx_list = [None] * MB
        total_ticks = MB + 4

        for tick in range(total_ticks):
            # Reverse pipeline order
            block._bwd_stage_pre_ops(tick - 4, fwd_ctx, ev_disp, dx_list)
            block._bwd_stage_experts(tick - 2, fwd_ctx, ev_comb, ev_exp)
            block._bwd_stage_post_ops(tick, fwd_ctx, grad_chunks, ev_post)

            block._bwd_stage_dispatch(tick - 3, fwd_ctx, ev_exp, ev_disp)
            block._bwd_stage_combine(tick - 1, fwd_ctx, ev_post, ev_comb)

        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])

        if not ctx.needs_input_grad[0]:
            return None, None

        return torch.cat(dx_list, dim=0), None


class PipelineMoEBlock(nn.Module):
    def __init__(self, original_layer, ep_config, rank, world_size, group, streams):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.streams = streams
        self.n_mb = ep_config.micro_batches
        self.hidden_dim = original_layer.hidden_size  # Adjust attr name if needed

        # 1. Replicated Modules
        self.input_layernorm = original_layer.input_layernorm
        self.self_attn = original_layer.self_attn
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.shared_experts = original_layer.mlp.shared_experts
        self.gate = original_layer.mlp.gate  # Router
        self.post_ops = nn.Identity()  # Placeholder if needed, usually handled in stage 5

        # 2. Sharded Experts
        all_experts = original_layer.mlp.experts
        n_routed = len(all_experts)
        self.num_local_experts = n_routed // world_size
        s = rank * self.num_local_experts
        self.local_experts = nn.ModuleList(
            [all_experts[i] for i in range(s, s + self.num_local_experts)]
        )

        self.cap_factor = ep_config.capacity_factor

    def forward(self, x):
        return MoEOverlapFunction.apply(x, self)

    # --- FORWARD STAGES ---

    def _fwd_stage_pre_ops(self, mb, ctx, chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            x = chunks[mb]
            # Save input for backward
            ctx[mb]["x_in"] = x

            # 1. Standard Attention Path
            residual = x
            x_norm = self.input_layernorm(x)
            # Assuming simple attention signature for compatibility.
            # In real integration, pass kwargs (mask, rope) via context or wrapper
            attn_out = self.self_attn(x_norm)
            x = residual + attn_out

            # 2. MoE Path Preparation
            residual_moe = x
            x_norm_moe = self.post_attention_layernorm(x)
            x_flat = x_norm_moe.view(-1, self.hidden_dim)

            # 3. Routing
            perm_in, perm_w, gather_idx, cap = self.gate(x_flat)

            # Save
            ctx[mb]["residual_moe"] = residual_moe
            ctx[mb]["x_norm_moe"] = x_norm_moe
            ctx[mb]["perm_in"] = perm_in.detach()
            ctx[mb]["perm_w"] = perm_w
            ctx[mb]["gather_idx"] = gather_idx
            ctx[mb]["cap"] = cap

        ev_signal[mb].record(stream)

    def _fwd_stage_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            perm_in = ctx[mb]["perm_in"]
            cap = ctx[mb]["cap"]
            # [World, LocalExperts*Cap, D]
            tokens = self.num_local_experts * cap
            send = perm_in.view(self.world_size, tokens, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["dispatched"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_experts(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            disp = ctx[mb]["dispatched"]
            cap = ctx[mb]["cap"]
            # [World, Local, Cap, D] -> [Local, World, Cap, D] -> [Local, World*Cap, D]
            inp = disp.view(self.world_size, self.num_local_experts, cap, self.hidden_dim)
            inp = inp.transpose(0, 1).reshape(self.num_local_experts, -1, self.hidden_dim)

            ctx[mb]["expert_input"] = inp.detach()  # Save for backward

            outs = []
            for i, expert in enumerate(self.local_experts):
                outs.append(expert(inp[i]))

            # [Local, World*Cap, D] -> [Local, World, Cap, D] -> [World, Local, Cap, D]
            stack = torch.stack(outs, dim=0)
            stack = stack.view(self.num_local_experts, self.world_size, cap, self.hidden_dim)
            ctx[mb]["expert_out"] = stack.transpose(0, 1).contiguous().view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["expert_out"]
            tokens = self.num_local_experts * ctx[mb]["cap"]
            send = send.view(self.world_size, tokens, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["combined"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_post_ops(self, mb, ctx, outputs, ev_wait, chunks):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            moe_out = ctx[mb]["combined"]
            perm_w = ctx[mb]["perm_w"]
            gather_idx = ctx[mb]["gather_idx"]
            residual = ctx[mb]["residual_moe"]

            # Weighted Un-permute
            weighted = moe_out * perm_w.unsqueeze(1)

            # Scatter Add
            # We need a buffer of shape [TotalTokens, Hidden]
            # Since residual is [Batch, Seq, Hidden], we view it flat
            res_flat = residual.view(-1, self.hidden_dim)
            buffer = torch.zeros_like(res_flat)

            valid = gather_idx != -1
            buffer.index_add_(0, gather_idx[valid], weighted[valid])

            # Shared Experts (Run on full input)
            # Assuming shared_experts takes the norm input
            shared_out = self.shared_experts(ctx[mb]["x_norm_moe"])

            # Final Add
            final = residual + buffer.view_as(residual) + shared_out
            outputs[mb] = final

            # Save for grad
            ctx[mb]["moe_out_for_grad"] = moe_out

    # --- BACKWARD STAGES ---

    def _bwd_stage_post_ops(self, mb, ctx, grad_chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            # Gradient arrives for 'final' output
            d_final = grad_chunks[mb]

            # Since final = residual + buffer + shared,
            # d_residual = d_final
            # d_buffer = d_final
            # d_shared = d_final

            ctx[mb]["d_final"] = d_final

            # Shared Expert Backward
            # Recompute graph or use saved tensors if differentiable?
            # Standard autograd handles this if we saved the graph, but here we manually split.
            # To simplify: We assume standard autograd for the SharedExpert module itself.
            # We trigger it by running forward on detached input with grad?
            # Correct approach for pipeline: Use autograd.grad on the specific ops.

            x_norm = ctx[mb]["x_norm_moe"].detach().requires_grad_(True)
            with torch.enable_grad():
                s_out = self.shared_experts(x_norm)

            torch.autograd.backward(s_out, d_final)
            ctx[mb]["d_x_norm_shared"] = x_norm.grad  # Gradient from shared expert path

        ev_signal[mb].record(stream)

    def _bwd_stage_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_final = ctx[mb]["d_final"].view(-1, self.hidden_dim)
            gather_idx = ctx[mb]["gather_idx"]
            perm_w = ctx[mb]["perm_w"]
            moe_out = ctx[mb]["moe_out_for_grad"]

            # 1. Gather Gradients (Backward of Scatter)
            d_weighted = torch.zeros(
                gather_idx.size(0), self.hidden_dim, device=d_final.device, dtype=d_final.dtype
            )
            valid = gather_idx != -1
            d_weighted[valid] = d_final[gather_idx[valid]]

            # 2. Gradient w.r.t Gate Weights
            # weighted = moe_out * perm_w
            # d_perm_w = sum(d_weighted * moe_out)
            d_perm_w = (d_weighted * moe_out).sum(dim=1)
            ctx[mb]["d_perm_w"] = d_perm_w

            # 3. Gradient w.r.t MOE Output
            # d_moe_out = d_weighted * perm_w
            d_moe_out = d_weighted * perm_w.unsqueeze(1)

            # 4. Reverse Combine (AllToAll)
            tokens = self.num_local_experts * ctx[mb]["cap"]
            send = d_moe_out.view(self.world_size, tokens, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["d_expert_out"] = recv.view(-1, self.hidden_dim)

        ev_signal[mb].record(stream)

    def _bwd_stage_experts(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_out = ctx[mb]["d_expert_out"]
            cap = ctx[mb]["cap"]
            inp = ctx[mb]["expert_input"]  # [Local, World*Cap, D]

            # Reverse Transpose logic from forward
            # Fwd: [W, L, C] -> [L, W, C] -> [L, W*C]
            # Bwd: [W, L, C] <- [L, W, C] <- [L, W*C]

            # Reshape d_out: [W, L, C] -> [L, W, C] -> [L, W*C]
            d_out_grouped = d_out.view(
                self.world_size, self.num_local_experts, cap, self.hidden_dim
            )
            d_out_grouped = (
                d_out_grouped.transpose(0, 1)
                .contiguous()
                .view(self.num_local_experts, -1, self.hidden_dim)
            )

            inp.requires_grad_(True)
            with torch.enable_grad():
                outs = []
                for i, expert in enumerate(self.local_experts):
                    outs.append(expert(inp[i]))
                grouped_out = torch.stack(outs)

            torch.autograd.backward(grouped_out, d_out_grouped)

            d_inp = inp.grad  # [Local, World*Cap, D]

            # Reverse Transpose for Dispatch
            # [L, WC] -> [L, W, C] -> [W, L, C]
            d_inp = d_inp.view(self.num_local_experts, self.world_size, cap, self.hidden_dim)
            ctx[mb]["d_dispatch"] = d_inp.transpose(0, 1).contiguous().view(-1, self.hidden_dim)

        ev_signal[mb].record(stream)

    def _bwd_stage_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["d_dispatch"]
            tokens = self.num_local_experts * ctx[mb]["cap"]
            send = send.view(self.world_size, tokens, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["d_perm_in"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _bwd_stage_pre_ops(self, mb, ctx, ev_wait, dx_list):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            # Gather gradients
            d_perm_in = ctx[mb]["d_perm_in"]  # From Dispatch
            d_perm_w = ctx[mb]["d_perm_w"]  # From Combine
            d_final = ctx[mb]["d_final"]  # From PostOps (residual path)
            d_x_norm_shared = ctx[mb]["d_x_norm_shared"]  # From PostOps (shared path)

            gather_idx = ctx[mb]["gather_idx"]
            x_norm_moe = ctx[mb]["x_norm_moe"]

            # 1. Reverse Permutation (d_perm_in -> d_x_norm)
            d_x_norm_routed = torch.zeros_like(x_norm_moe.view(-1, self.hidden_dim))
            valid = gather_idx != -1
            d_x_norm_routed.index_add_(0, gather_idx[valid], d_perm_in[valid])
            d_x_norm_routed = d_x_norm_routed.view_as(x_norm_moe)

            # 2. Backprop through Router (for d_perm_w)
            # We need to re-run router to get graph connecting x_norm to perm_w?
            # Or use autograd on x_norm?
            # Router output 'perm_w' depends on 'x_norm'.

            x_norm_grad = x_norm_moe.detach().requires_grad_(True)
            with torch.enable_grad():
                _, pw, _, _ = self.gate(x_norm_grad.view(-1, self.hidden_dim))

            torch.autograd.backward(pw, d_perm_w)
            d_x_norm_gate = x_norm_grad.grad

            # Total grad at x_norm_moe
            d_x_norm = d_x_norm_routed + d_x_norm_gate + d_x_norm_shared

            # 3. Backprop through Post-LN and Attention
            x_in = ctx[mb]["x_in"].detach().requires_grad_(True)
            with torch.enable_grad():
                # Replay Fwd Stage 1
                res = x_in
                x_n = self.input_layernorm(x_in)
                attn = self.self_attn(x_n)
                x_mid = res + attn
                x_out_norm = self.post_attention_layernorm(x_mid)

            # We have d_x_out_norm (which is d_x_norm)
            # We also have d_residual (which is d_final) acting on 'x_mid' (residual_moe)

            torch.autograd.backward((x_out_norm, x_mid), (d_x_norm, d_final))

            dx_list[mb] = x_in.grad


# ==========================================
# N-Block Tiny Model
# ==========================================
class TinyModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.moe.hidden_dim

        # Input Projection
        self.input_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Get the shared streams (Compute, Comm)
        self.streams: dict[Streams, torch.cuda.Stream] = get_ep_streams()
        self.blocks = nn.ModuleList()

        # Dynamic Block Construction
        # We implement the "Micro Batch Chain" where Block N computes Pre=Identity, Post=Attn(N+1).
        # Structure:
        # Block 0:   Pre=Attn(0), MoE(0), Post=Attn(1)
        # Block 1:   Pre=None,    MoE(1), Post=Attn(2)
        # ...
        # Block N-1: Pre=None,    MoE(N-1), Post=Linear(Out)

        for i in range(cfg.moe.n_blocks):
            # Pre-Op Logic:
            # Only the first block (i=0) needs to run its own Attention.
            # Subsequent blocks receive the output of Attn(i) which was computed in Block(i-1)'s Post-Op. # noqa
            if i == 0:
                pre_module = SelfAttention(self.hidden_dim, cfg.moe.num_heads)
            else:
                pre_module = None  # Becomes nn.Identity inside the block

            # Post-Op Logic:
            # Blocks 0 to N-2 compute the *next* block's Attention.
            # The final block (N-1) computes the final Linear layer (or Identity if no head).
            if i < cfg.moe.n_blocks - 1:
                post_module = SelfAttention(self.hidden_dim, cfg.moe.num_heads)
            else:
                # Final block post-op: Project to output or next stage
                post_module = nn.Linear(self.hidden_dim, self.hidden_dim)

            block = PipelineMoEBlock(
                cfg,
                group,
                block_name=f"B{i}",
                pre_op_module=pre_module,
                post_op_module=post_module,
                streams=self.streams,
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return x
