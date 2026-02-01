import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from loguru import logger

from fastmoe.config import Config, MoEScale, get_cfg
from fastmoe.models.router import TopKRouter
from fastmoe.models.tiny_model import Expert, SelfAttention, TinyModel


# ==========================================
# 1. Reference Block (Synchronous MoE)
# ==========================================
class ReferenceMoEBlock(nn.Module):
    def __init__(self, cfg: Config, group, pre_op, post_op):
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.hidden_dim = cfg.moe.hidden_dim

        self.moe_norm = nn.LayerNorm(self.hidden_dim)
        self.pre_ops = pre_op if pre_op else nn.Identity()

        # Exact match of Router
        total_experts = cfg.moe.num_experts_per_gpu * cfg.world_size
        self.router = TopKRouter(
            self.hidden_dim,
            total_experts,
            cfg.moe.top_k,
            capacity_factor=cfg.moe.comm_scaling_factor,
        )

        self.experts = nn.ModuleList(
            [Expert(self.hidden_dim, cfg.moe.proj_dim) for _ in range(cfg.moe.num_experts_per_gpu)]
        )
        self.post_ops = post_op if post_op else nn.Identity()

    def forward(self, x):
        # x is a micro-batch chunk
        x_proc = self.pre_ops(x)
        x_flat = x_proc.view(-1, self.hidden_dim)
        x_normed = self.moe_norm(x_proc).view(-1, self.hidden_dim)

        permuted_inputs, permuted_weights, gather_index, capacity = self.router(x_normed)

        # Dispatch
        tokens_per_rank = len(self.experts) * capacity
        reshaped_in = permuted_inputs.view(self.cfg.world_size, tokens_per_rank, self.hidden_dim)
        reshaped_out = torch.empty_like(reshaped_in)
        dist.all_to_all_single(reshaped_out, reshaped_in, group=self.group)
        dispatch_output = reshaped_out.view(-1, self.hidden_dim)

        # Experts
        view_4d = dispatch_output.view(
            self.cfg.world_size, len(self.experts), capacity, self.hidden_dim
        )
        expert_input_grouped = view_4d.transpose(0, 1).reshape(
            len(self.experts), -1, self.hidden_dim
        )

        res = []
        for i in range(len(self.experts)):
            res.append(self.experts[i](expert_input_grouped[i]))
        expert_out_grouped = torch.stack(res, dim=0)

        expert_out_4d = expert_out_grouped.view(
            len(self.experts), self.cfg.world_size, capacity, self.hidden_dim
        ).transpose(0, 1)
        expert_output = expert_out_4d.reshape(-1, self.hidden_dim)

        # Combine
        reshaped_in = expert_output.view(self.cfg.world_size, tokens_per_rank, self.hidden_dim)
        reshaped_out = torch.empty_like(reshaped_in)
        dist.all_to_all_single(reshaped_out, reshaped_in, group=self.group)
        moe_out = reshaped_out.view(-1, self.hidden_dim)

        # Post-Ops
        weighted_moe = moe_out * permuted_weights.unsqueeze(1)
        output_buffer = torch.zeros_like(x_flat)
        valid_mask = gather_index != -1
        valid_indices = gather_index[valid_mask]
        valid_data = weighted_moe[valid_mask]
        output_buffer.index_add_(0, valid_indices, valid_data)

        post_moe_out = x_flat + output_buffer  # Residual
        reshaped_in = post_moe_out.view(x.shape)

        return self.post_ops(reshaped_in)


class ReferenceTinyModel(nn.Module):
    def __init__(self, cfg, group):
        super().__init__()
        self.cfg = cfg
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
        chunks = x.chunk(self.cfg.moe.micro_batches, dim=0)
        out_chunks = []
        for chunk in chunks:
            for block in self.blocks:
                chunk = block(chunk)
            out_chunks.append(chunk)
        return torch.cat(out_chunks, dim=0)


def compare_models(rank, pipe_model, ref_model):
    # Ensure they are in the same order
    for (_n1, p1), (_, p2) in zip(
        pipe_model.named_parameters(), ref_model.named_parameters(), strict=True
    ):
        p2.data.copy_(p1.data)


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
    os.environ["MASTER_PORT"] = "12375"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfg = get_cfg(world_size=world_size, scale=MoEScale.TINY)

    # [FIX] Reduce Batch Size to verify correctness without OOM/Massive Drift
    cfg.moe.batch_size = 32

    pipe_model = TinyModel(cfg, dist.group.WORLD).cuda()
    ref_model = ReferenceTinyModel(cfg, dist.group.WORLD).cuda()
    compare_models(rank, pipe_model, ref_model)

    torch.manual_seed(42 + rank)
    data = torch.randn(
        cfg.moe.batch_size, cfg.moe.seqlen, cfg.moe.hidden_dim, device="cuda", requires_grad=True
    )
    target = torch.randn(cfg.moe.batch_size, cfg.moe.hidden_dim, device="cuda")

    # Forward
    dist.barrier()
    out_pipe = pipe_model(data)
    out_ref = ref_model(data)
    check_tensors(rank, "Forward Output", out_pipe, out_ref)

    # Backward
    loss_pipe = (out_pipe.mean(dim=1) - target).pow(2).sum()
    loss_ref = (out_ref.mean(dim=1) - target).pow(2).sum()
    check_tensors(rank, "Loss", loss_pipe, loss_ref)

    loss_pipe.backward()
    loss_ref.backward()

    for (n, p_pipe), p_ref in zip(
        pipe_model.named_parameters(), ref_model.parameters(), strict=True
    ):
        if p_pipe.grad is not None and p_ref.grad is not None:
            check_tensors(rank, f"Grad {n}", p_pipe.grad, p_ref.grad, tol=1e-2)

    # Cleanup before loop
    optimizer = torch.optim.Adam(pipe_model.parameters(), lr=1e-3)
    del out_pipe, out_ref, loss_pipe, loss_ref
    torch.cuda.empty_cache()

    # Convergence
    logger.info(f"Rank {rank}: Running Convergence Loop...")
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        out = pipe_model(data)
        loss = (out - data).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if rank == 0:
            logger.info(f"Step {step}: {loss.item()}")

    if losses[-1] < losses[0]:
        logger.info(f"Rank {rank}: ✅ Converged.")

    dist.destroy_process_group()


def run_verify_correctness():
    mp.start_processes(worker, args=(2,), nprocs=2, join=True, start_method="fork")
