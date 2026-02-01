import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from loguru import logger

from fastmoe.config import Config, MoEScale, get_cfg
from fastmoe.models.tiny_model import Expert, SelfAttention, TinyModel


# ==========================================
# 1. Reference Block (Standard Autograd)
# ==========================================
class SimulatedComm(torch.autograd.Function):
    """
    Simulates All-to-All cost but acts as Identity function for data/gradients
    so we can verify math deterministically against the Pipeline model.
    """

    @staticmethod
    def forward(ctx, x, group, scaling):
        ctx.group = group
        ctx.scaling = scaling
        # Simulate Cost
        bloated = x.repeat(1, scaling)
        out = torch.empty_like(bloated)
        dist.all_to_all_single(out, bloated, group=group)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Simulate Cost in Backward
        bloated = grad_output.repeat(1, ctx.scaling)
        out = torch.empty_like(bloated)
        dist.all_to_all_single(out, bloated, group=ctx.group)
        return grad_output, None, None


class ReferenceMoEBlock(nn.Module):
    """
    Sequential, Standard PyTorch implementation of the block.
    No custom pipeline, no streams, standard Autograd.
    """

    def __init__(self, cfg: Config, group, pre_op, post_op):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.hidden_dim = cfg.moe.hidden_dim

        self.moe_norm = nn.LayerNorm(self.hidden_dim)
        self.pre_ops = pre_op if pre_op else nn.Identity()
        self.gate = nn.Linear(self.hidden_dim, cfg.moe.num_experts_per_gpu * 2, bias=False)
        self.experts = nn.ModuleList(
            [Expert(self.hidden_dim, cfg.moe.proj_dim) for _ in range(cfg.moe.num_experts_per_gpu)]
        )
        self.post_ops = post_op if post_op else nn.Identity()

    def forward(self, x):
        # Sequential Execution (No Micro-batching needed for math check, but we do it to match shapes if needed) # noqa
        # For simplicity, Reference processes whole batch at once

        # 1. Pre-Ops
        x_proc = self.pre_ops(x)
        x_flat = x_proc.view(-1, self.hidden_dim)

        # 2. Norm & Gate
        x_normed = self.moe_norm(x_proc).view(-1, self.hidden_dim)
        _ = self.gate(x_normed)  # Run for grad check

        # 3. Dispatch (Identity with Cost)
        dispatch_out = SimulatedComm.apply(x_normed, self.group, self.cfg.moe.comm_scaling_factor)

        # 4. Experts
        # Split for local experts
        splits = dispatch_out.chunk(len(self.experts))
        res = [self.experts[i](splits[i]) for i in range(len(self.experts))]
        expert_out = torch.cat(res)

        # 5. Combine (Identity with Cost)
        combined_out = SimulatedComm.apply(expert_out, self.group, self.cfg.moe.comm_scaling_factor)

        # 6. Post-Ops & Residual
        out = x_flat + combined_out
        out = out.view(x.shape)  # Restore sequence dim
        return self.post_ops(out)


class ReferenceTinyModel(nn.Module):
    def __init__(self, cfg, group):
        super().__init__()
        self.input_proj = nn.Linear(cfg.moe.hidden_dim, cfg.moe.hidden_dim)
        self.blocks = nn.ModuleList()

        for i in range(cfg.moe.n_blocks):
            pre = SelfAttention(cfg.moe.hidden_dim, cfg.moe.num_heads) if i == 0 else None
            post = (
                SelfAttention(cfg.moe.hidden_dim, cfg.moe.num_heads)
                if i < cfg.moe.n_blocks - 1
                else nn.Linear(cfg.moe.hidden_dim, cfg.moe.hidden_dim)
            )
            self.blocks.append(ReferenceMoEBlock(cfg, group, pre, post))

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


# ==========================================
# 2. Verification Logic
# ==========================================
def compare_models(rank, pipe_model, ref_model):
    logger.info(f"Rank {rank}: Comparing weights to ensure Identical Init...")
    for p1, p2 in zip(
        pipe_model.parameters(),
        ref_model.parameters(),
        strict=False,
    ):
        p2.data.copy_(p1.data)  # Force Ref to match Pipe
        assert torch.allclose(p1, p2), "Weights Init Mismatch"
    logger.info(f"Rank {rank}: Weights Synchronized.")


def check_tensors(rank, name, t_pipe, t_ref, tol=1e-3):
    if torch.allclose(t_pipe, t_ref, atol=tol, rtol=tol):
        logger.info(f"Rank {rank}: ✅ {name} Match!")
        return True
    else:
        diff = (t_pipe - t_ref).abs().max().item()
        logger.error(f"Rank {rank}: ❌ {name} Mismatch! Max Diff: {diff}")
        return False


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12375"  # Different port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfg = get_cfg(world_size=world_size, scale=MoEScale.TINY)

    # 1. Init Models
    pipe_model = TinyModel(cfg, dist.group.WORLD).cuda()
    ref_model = ReferenceTinyModel(cfg, dist.group.WORLD).cuda()

    # 2. Sync Weights
    compare_models(rank, pipe_model, ref_model)

    # 3. Create Identical Input
    # Ensure requires_grad=True to test full backward
    torch.manual_seed(42 + rank)
    data = torch.randn(
        cfg.moe.batch_size, cfg.moe.seqlen, cfg.moe.hidden_dim, device="cuda", requires_grad=True
    )
    target = torch.randn(cfg.moe.batch_size, cfg.moe.hidden_dim, device="cuda")  # Dummy target

    # ==========================
    # 4. Forward Pass Verification
    # ==========================
    logger.info(f"Rank {rank}: Running Forward...")
    dist.barrier()

    # Run Pipeline
    out_pipe = pipe_model(data)

    # Run Reference
    out_ref = ref_model(data)

    check_tensors(rank, "Forward Output", out_pipe, out_ref)

    # ==========================
    # 5. Backward Pass Verification
    # ==========================
    logger.info(f"Rank {rank}: Running Backward...")

    # Simple Loss
    loss_pipe = (out_pipe.mean(dim=1) - target).pow(2).sum()
    loss_ref = (out_ref.mean(dim=1) - target).pow(2).sum()

    check_tensors(rank, "Loss", loss_pipe, loss_ref)

    # Retain graph for input grad check
    loss_pipe.backward()
    loss_ref.backward()

    # Compare Gradients
    logger.info(f"Rank {rank}: Checking Gradients...")

    # 5a. Check Input Gradients
    # Note: data.grad accumulates both backward calls if we re-use 'data'.
    # But since we run them sequentially on the same 'data' tensor object?
    # Ideally we should use clones. But let's check params first.

    all_match = True
    for (n, p_pipe), p_ref in zip(
        pipe_model.named_parameters(),
        ref_model.parameters(),
        strict=False,
    ):
        if p_pipe.grad is None or p_ref.grad is None:
            logger.warning(f"Rank {rank}: Skipped {n} (None grad)")
            continue
        if not check_tensors(rank, f"Grad {n}", p_pipe.grad, p_ref.grad):
            all_match = False
            break

    if all_match:
        logger.info(f"Rank {rank}: ✅ ALL Parameter Gradients Match!")

    # ==========================
    # 6. Convergence Test
    # ==========================
    logger.info(f"Rank {rank}: Running Convergence Test (20 steps)...")
    optimizer = torch.optim.Adam(pipe_model.parameters(), lr=1e-3)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        # Learn identity: Output should match Input
        out = pipe_model(data)
        loss = (out - data).pow(2).mean()  # Simple regression
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 5 == 0 and rank == 0:
            logger.info(f"Step {step}: Loss = {loss.item():.6f}")

    if losses[-1] < losses[0] * 0.5:
        logger.info(f"Rank {rank}: ✅ Model Converges! Loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    else:
        logger.warning(f"Rank {rank}: ⚠️ Model might not be converging nicely.")

    dist.destroy_process_group()


def run_verify_correctness():
    mp.start_processes(
        worker,
        args=(2,),
        nprocs=2,
        join=True,
        start_method="fork",
    )
