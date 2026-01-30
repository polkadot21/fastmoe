import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.profiler import record_function

from fastmoe.comm import Streams, get_ep_streams
from fastmoe.config import Config


# ==========================================
# Modules
# ==========================================
class SelfAttention(nn.Module):
    """
    Standard Multi-Head Attention to simulate realistic compute workloads.
    """

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
        # Input Shape: [Batch, SeqLen, HiddenDim]
        B, S, D = x.shape
        residual = x
        x = self.norm(x)

        # Projections & Reshape for Multi-head
        # View: [B, S, NumHeads, HeadDim] -> Transpose: [B, NumHeads, S, HeadDim]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reassemble heads
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out + residual


class Expert(nn.Module):
    """
    A single Expert (FFN). In production (DeepSeek/Mixtral), this would often be a SwiGLU.
    """

    def __init__(self, dim: int, proj_dim: int) -> None:
        super().__init__()

        # Assertion ensures we adhere to expansion ratios typical in Transformers (e.g. 4x or 8/3x)
        assert proj_dim > dim, "Projection Dimensions should be larger than dimension"

        self.fc1 = nn.Linear(dim, proj_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Num_Tokens_Assigned_To_This_Expert, HiddenDim]
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# Configurable PipeLine Block (Shared Streams)
# ==========================================
class MoEOverlapFunction(Function):
    """
    Orchestrates the Lancet 5-stage pipeline for both Forward and Backward passes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, block: "PipelineMoEBlock") -> torch.Tensor:
        #

        # Setup Context
        ctx.block = block

        # Micro-batching
        chunks = x.chunk(block.cfg.moe.micro_batches, dim=0)

        # Buffers & Events
        # fwd_ctx: Stores tensors needed for the Backward pass.
        fwd_ctx = [{} for _ in range(block.cfg.moe.micro_batches)]
        outputs = [None] * block.cfg.moe.micro_batches

        # Events for Producer-Consumer sync between Compute and Comm streams
        ev_pre = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_disp = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_exp = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_comb = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]

        total_ticks = block.cfg.moe.micro_batches + 4

        # Pipeline Loop (Forward Wavefront)
        for tick in range(total_ticks):
            mb_post = tick - 4
            mb_comb = tick - 3
            mb_exp = tick - 2
            mb_disp = tick - 1
            mb_pre = tick

            # Delegate to Block Methods
            # We call them in reverse logic order (Post->Pre) to submit work to streams
            block._fwd_stage_post_ops(mb_post, fwd_ctx, outputs, ev_comb, chunks)
            block._fwd_stage_combine(mb_comb, fwd_ctx, ev_exp, ev_comb)
            block._fwd_stage_experts(mb_exp, fwd_ctx, ev_disp, ev_exp)
            block._fwd_stage_dispatch(mb_disp, fwd_ctx, ev_pre, ev_disp)
            block._fwd_stage_pre_ops(mb_pre, fwd_ctx, chunks, ev_pre)

        # Final Sync & Save
        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
        ctx.fwd_ctx = fwd_ctx
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        #
        block: PipelineMoEBlock = ctx.block
        fwd_ctx = ctx.fwd_ctx

        # Chunk Gradients
        grad_chunks = grad_output.chunk(block.cfg.moe.micro_batches, dim=0)

        # Events for Backward Sync (Inverted Dependencies)
        ev_post_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_comb_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_exp_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]
        ev_disp_bw = [torch.cuda.Event() for _ in range(block.cfg.moe.micro_batches)]

        dx_list = [None] * block.cfg.moe.micro_batches
        total_ticks = block.cfg.moe.micro_batches + 4

        # Reverse Pipeline Loop
        for tick in range(total_ticks):
            # Reverse Scheduling offsets
            mb_post = tick
            mb_comb = tick - 1
            mb_exp = tick - 2
            mb_disp = tick - 3
            mb_pre = tick - 4

            # Delegate to Block Methods (Reverse Order Logic)
            block._bwd_stage_post_ops(mb_post, fwd_ctx, grad_chunks, ev_post_bw)
            block._bwd_stage_combine(mb_comb, fwd_ctx, ev_post_bw, ev_comb_bw)
            block._bwd_stage_experts(mb_exp, fwd_ctx, ev_comb_bw, ev_exp_bw)
            block._bwd_stage_dispatch(mb_disp, fwd_ctx, ev_exp_bw, ev_disp_bw)
            block._bwd_stage_pre_ops(mb_pre, fwd_ctx, ev_disp_bw, dx_list)

        # Final Sync & Return
        torch.cuda.current_stream().wait_stream(block.streams[Streams.COMPUTE])
        return torch.cat(dx_list, dim=0), None


# ==========================================
# Configurable PipeLine Block (Logic Owner)
# ==========================================
class PipelineMoEBlock(nn.Module):
    def __init__(
        self,
        cfg: Config,
        group: dist.ProcessGroup,
        block_name: str,
        pre_op_module: SelfAttention | None,
        post_op_module: SelfAttention | nn.Linear | None,
        streams: dict[Streams, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        self.cfg: Config = cfg
        self.block_name = block_name
        self.hidden_dim = cfg.moe.hidden_dim
        self.num_local_experts = cfg.moe.num_experts_per_gpu
        self.group = group
        self.streams = streams

        # Components
        self.moe_norm = nn.LayerNorm(self.hidden_dim)
        self.pre_ops = pre_op_module if pre_op_module else nn.Identity()
        self.gate = nn.Linear(
            self.hidden_dim, cfg.moe.num_experts_per_gpu * cfg.world_size, bias=False
        )
        self.experts = nn.ModuleList(
            [Expert(self.hidden_dim, cfg.moe.proj_dim) for _ in range(self.num_local_experts)]
        )
        self.post_ops = post_op_module if post_op_module else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MoEOverlapFunction.apply(x, self)

    # =========================================================================
    # FORWARD STAGES
    # =========================================================================

    def _fwd_stage_pre_ops(self, mb_idx, ctx, chunks, ev_signal):
        """Stage 1 (Fwd): Pre-Ops + Gating [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Pre_MB{mb_idx}"
                with record_function(label):
                    x_mb = chunks[mb_idx]

                    with torch.enable_grad():
                        x_proc = self.pre_ops(x_mb)
                        x_flat = x_proc.view(-1, self.hidden_dim)
                        x_normed = self.moe_norm(x_proc).view(-1, self.hidden_dim)
                        # Gating
                        logits = self.gate(x_normed)
                        _, _ = torch.topk(logits, k=self.cfg.moe.top_k, dim=1)

                    # Save for next stages
                    # DETACH here to break the graph between stages.
                    # If we don't, Backward Post-Ops will try to backprop through this
                    # link immediately, crashing the pipeline.
                    ctx[mb_idx]["normed_input_flat"] = x_normed.detach()
                    ctx[mb_idx]["gated_input"] = x_flat.detach()

                    # Save Attached inputs for Backward Re-run
                    ctx[mb_idx]["input_pre"] = x_mb

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_dispatch(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 2 (Fwd): Dispatch [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Dispatch_MB{mb_idx}"
                with record_function(label):
                    normed_in = buf["normed_input_flat"]

                    # Simulation Bloat
                    bloated_in = normed_in.repeat(1, self.cfg.moe.comm_scaling_factor)
                    bloated_out = torch.empty_like(bloated_in)
                    dist.all_to_all_single(
                        bloated_out, bloated_in, group=self.group, async_op=False
                    )

                    # New tensor -> implicitly detached.
                    buf["dispatch_output"] = torch.empty_like(normed_in)

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_experts(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 3 (Fwd): Experts [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Experts_MB{mb_idx}"
                with record_function(label):
                    disp_out = buf["dispatch_output"]
                    # Save for Backward
                    # Store detached input for re-computation
                    buf["input_experts"] = disp_out.detach()

                    with torch.enable_grad():
                        splits = disp_out.chunk(self.num_local_experts)
                        res = [self.experts[i](splits[i]) for i in range(self.num_local_experts)]
                        expert_out = torch.cat(res)

                    buf["expert_output"] = expert_out.detach()

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_combine(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 4 (Fwd): Combine [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Combine_MB{mb_idx}"
                with record_function(label):
                    expert_out = buf["expert_output"]

                    bloated_in = expert_out.repeat(1, self.cfg.moe.comm_scaling_factor)
                    bloated_out = torch.empty_like(bloated_in)
                    dist.all_to_all_single(
                        bloated_out, bloated_in, group=self.group, async_op=False
                    )

                    # Implicitly detached
                    buf["combined_output"] = torch.empty_like(expert_out)

            ev_signal[mb_idx].record(stream)

    def _fwd_stage_post_ops(self, mb_idx, ctx, outputs, ev_wait, chunks):
        """Stage 5 (Fwd): Post-Ops [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Fwd_Post_MB{mb_idx}"
                with record_function(label):
                    moe_out = buf["combined_output"]
                    residual = buf["gated_input"]

                    # Compute
                    # Inputs are detached, so this builds a local, isolated graph
                    post_moe_out = residual + moe_out
                    B_mb = chunks[mb_idx].size(0)
                    reshaped_in = post_moe_out.view(B_mb, -1, self.hidden_dim)

                    with torch.enable_grad():
                        out = self.post_ops(reshaped_in)

                    outputs[mb_idx] = out
                    # Save for Backward
                    buf["input_post"] = reshaped_in

    # =========================================================================
    # BACKWARD STAGES
    # =========================================================================

    def _bwd_stage_post_ops(self, mb_idx, ctx, grad_chunks, ev_signal):
        """Stage 1 (Bwd): Grad Post-Ops [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Post_MB{mb_idx}"
                with record_function(label):
                    # We need to re-attach the input to a graph to compute grads
                    inp = buf["input_post"].detach().requires_grad_(True)

                    with torch.enable_grad():
                        out = self.post_ops(inp)

                    # Compute gradients using explicit torch.autograd.grad
                    # This ensures we don't accidentally traverse a leaked graph
                    # and allows us to manually accumulate into param.grad
                    grads = torch.autograd.grad(
                        outputs=(out,),
                        inputs=(inp,) + tuple(self.post_ops.parameters()),
                        grad_outputs=(grad_chunks[mb_idx],),
                    )

                    d_inp = grads[0]
                    d_params = grads[1:]

                    # Manual Accumulation into .grad
                    for p, g in zip(self.post_ops.parameters(), d_params, strict=False):
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                    # Split gradients: d_inp = d(Residual + MoE) -> d_Res + d_MoE
                    buf["grad_combined"] = d_inp
                    buf["grad_residual"] = d_inp

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_combine(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 2 (Bwd): Grad Combine [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Combine_MB{mb_idx}"
                with record_function(label):
                    d_moe = buf["grad_combined"]

                    # Communication layers operate on flattened tokens.
                    d_moe_flat = d_moe.view(-1, self.hidden_dim)

                    # Simulation Bloat
                    bloated_in = d_moe_flat.repeat(1, self.cfg.moe.comm_scaling_factor)
                    bloated_out = torch.empty_like(bloated_in)
                    dist.all_to_all_single(
                        bloated_out, bloated_in, group=self.group, async_op=False
                    )

                    # We must ensure the output shape matches what Experts expect (2D)
                    buf["grad_expert_out"] = torch.empty_like(d_moe_flat)

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_experts(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 3 (Bwd): Grad Experts [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Experts_MB{mb_idx}"
                with record_function(label):
                    d_expert = buf["grad_expert_out"]
                    inp = buf["input_experts"].detach().requires_grad_(True)

                    with torch.enable_grad():
                        splits = inp.chunk(self.num_local_experts)
                        res = [self.experts[i](splits[i]) for i in range(self.num_local_experts)]
                        out = torch.cat(res)

                    grads = torch.autograd.grad(
                        outputs=(out,),
                        inputs=(inp,) + tuple(self.experts.parameters()),
                        grad_outputs=(d_expert,),
                    )

                    d_inp = grads[0]
                    d_params = grads[1:]

                    for p, g in zip(self.experts.parameters(), d_params, strict=False):
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                    buf["grad_dispatch_out"] = d_inp

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_dispatch(self, mb_idx, ctx, ev_wait, ev_signal):
        """Stage 4 (Bwd): Grad Dispatch [COMM STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMM]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Dispatch_MB{mb_idx}"
                with record_function(label):
                    d_disp = buf["grad_dispatch_out"]

                    bloated_in = d_disp.repeat(1, self.cfg.moe.comm_scaling_factor)
                    bloated_out = torch.empty_like(bloated_in)
                    dist.all_to_all_single(
                        bloated_out, bloated_in, group=self.group, async_op=False
                    )

                    buf["grad_normed"] = torch.empty_like(d_disp)

            ev_signal[mb_idx].record(stream)

    def _bwd_stage_pre_ops(self, mb_idx, ctx, ev_wait, dx_list):
        """Stage 5 (Bwd): Grad Pre-Ops [COMPUTE STREAM]"""
        if 0 <= mb_idx < self.cfg.moe.micro_batches:
            stream = self.streams[Streams.COMPUTE]
            buf = ctx[mb_idx]

            stream.wait_event(ev_wait[mb_idx])
            with torch.cuda.stream(stream):
                label = f"{self.block_name}_Bwd_Pre_MB{mb_idx}"
                with record_function(label):
                    d_normed = buf["grad_normed"]
                    d_resid = buf["grad_residual"]
                    x_in = buf["input_pre"]

                    with torch.enable_grad():
                        x_proc = self.pre_ops(x_in)
                        x_flat = x_proc.view(-1, self.hidden_dim)
                        x_normed = self.moe_norm(x_proc).view(-1, self.hidden_dim)

                    d_resid_flat = d_resid.view(-1, self.hidden_dim)

                    # Add allow_unused=True because moe_norm is not used for x_flat
                    grads = torch.autograd.grad(
                        outputs=(x_flat, x_normed),
                        grad_outputs=(d_resid_flat, d_normed),
                        inputs=(x_in,)
                        + tuple(self.pre_ops.parameters())
                        + tuple(self.moe_norm.parameters()),
                        allow_unused=True,
                    )

                    d_x = grads[0]
                    d_params = grads[1:]

                    # Accumulate Params with None check
                    all_params = list(self.pre_ops.parameters()) + list(self.moe_norm.parameters())
                    for p, g in zip(all_params, d_params, strict=False):
                        if g is None:  # [FIX] Handle unused gradients
                            continue
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad += g

                    dx_list[mb_idx] = d_x


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

        # Get the shared streams (Compute, Comm, Expert)
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
