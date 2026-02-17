import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import Config
from fastmoe.models.router import TopKRouter


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


# ==========================================
# Configurable PipeLine Block
# ==========================================
class PipelineMoEBlock(nn.Module):
    def __init__(
        self,
        cfg: Config,
        group: dist.ProcessGroup,
        block_name: str,
        pre_op_module: nn.Module | None,
        post_op_module: nn.Module | None,
        streams: dict[Streams, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_name = block_name
        self.group = group
        self.streams = streams

        # --- Config Mapping ---
        # Map 'cfg' (Test) values to attributes expected by the Pipeline logic
        self.n_mb = cfg.moe.micro_batches
        self.hidden_dim = cfg.moe.hidden_dim
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # --- 1. Replicated Modules (Built from scratch for Testing) ---
        self.input_layernorm = nn.LayerNorm(self.hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_dim)

        # Pre/Post Ops (Attention / Linear) provided by TinyModel
        self.pre_ops = pre_op_module if pre_op_module else nn.Identity()
        self.post_ops = post_op_module if post_op_module else nn.Identity()

        # Shared Expert (Dummy implementation for test parity)
        # We use a simple Linear to mimic the shared expert
        self.shared_experts = nn.Linear(self.hidden_dim, self.hidden_dim)

        # --- 2. Router ---
        total_experts = cfg.moe.num_experts_per_gpu * self.world_size
        self.gate = TopKRouter(
            hidden_dim=self.hidden_dim,
            num_total_experts=total_experts,
            top_k=cfg.moe.top_k,
            capacity_factor=cfg.moe.comm_scaling_factor,
        )

        # --- 3. Sharded Experts ---
        # In test mode, we only initialize the LOCAL experts for this rank.
        self.num_local_experts = cfg.moe.num_experts_per_gpu
        self.local_experts = nn.ModuleList(
            [Expert(self.hidden_dim, cfg.moe.proj_dim) for _ in range(self.num_local_experts)]
        )

        self.cap_factor = cfg.moe.comm_scaling_factor

    # Bridge between TinyModel's expected interface and the Production Logic
    @property
    def self_attn(self):
        """Alias pre_ops as self_attn to match Production Logic names."""
        return self.pre_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MoEOverlapFunction.apply(x, self)

    # =========================================================================
    # FORWARD STAGES (Identical to Production)
    # =========================================================================

    def _fwd_stage_pre_ops(self, mb, ctx, chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            x = chunks[mb]
            ctx[mb]["x_in"] = x

            # 1. Attn Path (Simulated by Pre-Ops)
            residual = x
            x_norm = self.input_layernorm(x)

            # In TinyModel, pre_ops IS the attention mechanism
            attn_out = self.pre_ops(x_norm)

            # If pre_ops is Identity, attn_out is x_norm.
            # TinyModel logic might differ slightly from GigaChat here,
            # but we follow the structure: Residual + Attn
            x_mid = residual + attn_out

            # 2. MoE Path
            residual_moe = x_mid
            x_norm_moe = self.post_attention_layernorm(x_mid)
            x_flat = x_norm_moe.view(-1, self.hidden_dim)

            # 3. Routing
            perm_in, perm_w, gather_idx, cap = self.gate(x_flat)

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
            # Reshape [World, Local, Cap] -> [Local, World, Cap]
            inp = disp.view(self.world_size, self.num_local_experts, cap, self.hidden_dim)
            inp = inp.transpose(0, 1).reshape(self.num_local_experts, -1, self.hidden_dim)
            ctx[mb]["expert_input"] = inp.detach()

            outs = [expert(inp[i]) for i, expert in enumerate(self.local_experts)]

            stack = torch.stack(outs, dim=0)
            ctx[mb]["expert_out"] = (
                stack.view(self.num_local_experts, self.world_size, cap, self.hidden_dim)
                .transpose(0, 1)
                .contiguous()
                .view(-1, self.hidden_dim)
            )
        ev_signal[mb].record(stream)

    def _fwd_stage_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["expert_out"].view(self.world_size, -1, self.hidden_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            ctx[mb]["combined"] = recv.view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _fwd_stage_post_ops(self, mb, ctx, outputs, ev_wait, chunks):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]

        # Wait for Combine Stage (on COMM stream) to finish
        stream.wait_event(ev_wait[mb])

        with torch.cuda.stream(stream):
            moe_out = ctx[mb]["combined"]
            perm_w = ctx[mb]["perm_w"]
            gather_idx = ctx[mb]["gather_idx"]
            residual = ctx[mb]["residual_moe"]

            # Weighted Un-permute
            weighted = moe_out * perm_w.unsqueeze(1)

            # Scatter Add
            res_flat = residual.view(-1, self.hidden_dim)
            buffer = torch.zeros_like(res_flat)

            valid = gather_idx != -1
            buffer.index_add_(0, gather_idx[valid], weighted[valid])

            # Shared Experts
            shared_out = self.shared_experts(ctx[mb]["x_norm_moe"])

            # Final MoE Block Output
            moe_final = residual + buffer.view_as(residual) + shared_out

            # In TinyModel, we have an explicit Post-Op (Next Block's Attention or Final Linear)
            # We run it here.
            out = self.post_ops(moe_final)
            outputs[mb] = out

            # Save for grad
            ctx[mb]["moe_out_grad_src"] = moe_out
            ctx[mb]["moe_final_input"] = moe_final  # Input to post_ops

    # =========================================================================
    # BACKWARD STAGES
    # =========================================================================

    def _bwd_stage_post_ops(self, mb, ctx, grad_chunks, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMPUTE]
        with torch.cuda.stream(stream):
            # Gradient comes from Next Block or Loss
            d_out = grad_chunks[mb]

            # 1. Backprop through Post Ops (Linear/Attn)
            # We need the input that went INTO post_ops
            post_in = ctx[mb]["moe_final_input"].detach().requires_grad_(True)

            with torch.enable_grad():
                post_out = self.post_ops(post_in)

            torch.autograd.backward(post_out, d_out)

            # d_final is the gradient at the output of the MoE block (before post_ops)
            d_final = post_in.grad
            ctx[mb]["d_final"] = d_final

            # 2. Shared Experts Grads
            x_norm = ctx[mb]["x_norm_moe"].detach().requires_grad_(True)
            with torch.enable_grad():
                s_out = self.shared_experts(x_norm)
            torch.autograd.backward(s_out, d_final)
            ctx[mb]["d_x_norm_shared"] = x_norm.grad
        ev_signal[mb].record(stream)

    def _bwd_stage_combine(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            d_final = ctx[mb]["d_final"].view(-1, self.hidden_dim)
            gather_idx = ctx[mb]["gather_idx"]
            moe_out = ctx[mb]["moe_out_grad_src"]

            d_weighted = torch.zeros_like(moe_out)
            valid = gather_idx != -1
            d_weighted[valid] = d_final[gather_idx[valid]]

            # Grads for Router Weights
            d_perm_w = (d_weighted * moe_out).sum(dim=1)
            ctx[mb]["d_perm_w"] = d_perm_w

            # Grads for Expert Output
            d_moe_out = d_weighted * ctx[mb]["perm_w"].unsqueeze(1)

            # AllToAll
            send = d_moe_out.view(self.world_size, -1, self.hidden_dim)
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
            inp = ctx[mb]["expert_input"]

            d_out_grp = d_out.view(self.world_size, self.num_local_experts, cap, self.hidden_dim)
            d_out_grp = (
                d_out_grp.transpose(0, 1)
                .contiguous()
                .view(self.num_local_experts, -1, self.hidden_dim)
            )

            inp.requires_grad_(True)
            with torch.enable_grad():
                outs = [expert(inp[i]) for i, expert in enumerate(self.local_experts)]
                grouped = torch.stack(outs)
            torch.autograd.backward(grouped, d_out_grp)

            d_inp = inp.grad
            d_inp = d_inp.view(self.num_local_experts, self.world_size, cap, self.hidden_dim)
            ctx[mb]["d_dispatch"] = d_inp.transpose(0, 1).contiguous().view(-1, self.hidden_dim)
        ev_signal[mb].record(stream)

    def _bwd_stage_dispatch(self, mb, ctx, ev_wait, ev_signal):
        if not (0 <= mb < self.n_mb):
            return
        stream = self.streams[Streams.COMM]
        stream.wait_event(ev_wait[mb])
        with torch.cuda.stream(stream):
            send = ctx[mb]["d_dispatch"].view(self.world_size, -1, self.hidden_dim)
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
            d_perm_in = ctx[mb]["d_perm_in"]
            d_perm_w = ctx[mb]["d_perm_w"]
            d_final = ctx[mb]["d_final"]
            d_x_norm_shared = ctx[mb]["d_x_norm_shared"]

            gather_idx = ctx[mb]["gather_idx"]
            x_norm_moe = ctx[mb]["x_norm_moe"]

            # 1. Reverse Permutation
            d_x_norm_routed = torch.zeros_like(x_norm_moe.view(-1, self.hidden_dim))
            valid = gather_idx != -1
            d_x_norm_routed.index_add_(0, gather_idx[valid], d_perm_in[valid])
            d_x_norm_routed = d_x_norm_routed.view_as(x_norm_moe)

            # 2. Router Backward
            x_norm_grad = x_norm_moe.detach().requires_grad_(True)
            with torch.enable_grad():
                _, pw, _, _ = self.gate(x_norm_grad.view(-1, self.hidden_dim))
            torch.autograd.backward(pw, d_perm_w)
            d_x_norm_gate = x_norm_grad.grad

            # Total grad at Post-Attn Norm
            d_x_norm = d_x_norm_routed + d_x_norm_gate + d_x_norm_shared

            # 3. Backprop through Post-LN and Attention
            x_in = ctx[mb]["x_in"].detach().requires_grad_(True)
            with torch.enable_grad():
                # Replay Fwd Stage 1
                res = x_in
                x_n = self.input_layernorm(x_in)
                attn = self.pre_ops(x_n)  # pre_ops is SelfAttn
                x_mid = res + attn
                x_out_norm = self.post_attention_layernorm(x_mid)

            # d_final flows to x_mid (residual). d_x_norm flows to x_out_norm.
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
