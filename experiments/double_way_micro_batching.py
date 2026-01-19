###
# Cell 1 — Imports + distributed init helpers
###

import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule


def ddp_init():
    """
    Works with:
      - torchrun (preferred)
      - mp.spawn inside a notebook (see later cell)
    """
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Good defaults for H100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return rank, world_size, local_rank


def barrier():
    if dist.is_initialized():
        dist.barrier()


def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def r0log(msg: str):
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(msg, flush=True)


def sync():
    torch.cuda.synchronize()


os.environ["PYTHONUNBUFFERED"] = "1"

###
# Cell 2 — Async all-to-all wrapper (keeps Work handle alive)
###


import inspect  # noqa


def _all_to_all_single_async(out, inp, *, group, in_splits, out_splits):
    """
    Compatibility shim for different PyTorch arg names:
      - some versions use input_split_sizes/output_split_sizes
      - others use in_split_sizes/out_split_sizes
    """
    sig = inspect.signature(dist.all_to_all_single)
    params = sig.parameters

    kwargs = {"group": group, "async_op": True}

    if "input_split_sizes" in params and "output_split_sizes" in params:
        kwargs["input_split_sizes"] = in_splits
        kwargs["output_split_sizes"] = out_splits
    else:
        # Older/alternate naming
        kwargs["in_split_sizes"] = in_splits
        kwargs["out_split_sizes"] = out_splits

    return dist.all_to_all_single(out, inp, **kwargs)


def all_to_all_single_async(input_tensor: torch.Tensor, in_splits, out_splits, group=None):
    """
    Variable-split all_to_all_single with async_op=True.
    Returns output tensor. Work handle attached as output._work.
    """
    assert input_tensor.is_cuda
    out0 = torch.empty(
        (sum(out_splits),) + input_tensor.shape[1:],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    work = _all_to_all_single_async(
        out0, input_tensor, group=group, in_splits=in_splits, out_splits=out_splits
    )
    out0._work = work
    return out0


def exchange_counts_async(send_counts_i32: torch.Tensor, group=None):
    """
    send_counts_i32: [world_size] int32 on GPU
    returns recv_counts_i32: [world_size] int32 on GPU (async), plus work handle on tensor
    """
    ws = send_counts_i32.numel()
    recv = torch.empty_like(send_counts_i32)
    ones = [1] * ws
    work = _all_to_all_single_async(
        recv, send_counts_i32, group=group, in_splits=ones, out_splits=ones
    )
    recv._work = work
    return recv


###
# Cell 3 — Router + Expert MLP
###


class Top2Router(nn.Module):
    def __init__(self, d_model: int, num_experts_global: int):
        super().__init__()
        self.w = nn.Linear(d_model, num_experts_global, bias=False)

    def forward(self, x):  # x: [T, D]
        scores = self.w(x)  # [T, E]
        probs = F.softmax(scores, dim=-1)  # [T, E]
        topv, topi = torch.topk(probs, k=2, dim=-1)  # [T, 2], [T, 2]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        return topi, topv


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


###
# Cell 4 — DeepEP-style 2-way microbatch Expert Parallel MoE
###


@dataclass
class DispatchMeta:
    # Origin-rank side (needed to assemble after return)
    send_counts: list
    recv_counts: list
    orig_idx_send_order: torch.Tensor  # [Nsend] int32 on GPU


@dataclass
class RecvPack:
    # Expert-rank side (received from others)
    x: torch.Tensor  # [Nrecv, D]
    expert_id: torch.Tensor  # [Nrecv] int32 (global expert id)
    orig_idx: torch.Tensor  # [Nrecv] int32 (token index within microbatch on origin)
    gate_w: torch.Tensor  # [Nrecv] float (weight)


class DeepEPMoE(nn.Module):
    """
    Expert Parallel MoE with DeepEP-style 2-way microbatch overlap.

    EP mapping:
      - global experts = ep_world_size * local_experts
      - rank r owns experts [r*local_experts : (r+1)*local_experts)

    This is *functional* and meant to be easy to profile & extend.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        local_experts: int,
        ep_group=None,
    ):
        super().__init__()
        assert dist.is_initialized()
        self.ep_group = ep_group
        self.ep_world = dist.get_world_size(group=ep_group)
        self.ep_rank = dist.get_rank(group=ep_group)

        self.local_experts = local_experts
        self.num_experts_global = self.ep_world * local_experts

        self.router = Top2Router(d_model, self.num_experts_global)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_hidden) for _ in range(local_experts)])

        # Streams: separate comm & compute to overlap
        self.dispatch_stream = torch.cuda.Stream(priority=-1)
        self.compute_stream = torch.cuda.Stream(priority=0)  # could also use default
        self.combine_stream = torch.cuda.Stream(priority=-1)
        self.assemble_stream = torch.cuda.Stream(priority=0)

    def _dispatch_one_microbatch(self, x_mb: torch.Tensor, mb_tag: str):
        """
        x_mb: [Tmb, D]
        Returns:
          recv_pack (expert-side inputs),
          dispatch_meta (origin-side metadata)
        """
        device = x_mb.device
        Tmb, D = x_mb.shape
        topi, topw = self.router(x_mb)  # [Tmb,2], [Tmb,2]

        # Replicate tokens for top-2
        x_rep = x_mb.unsqueeze(1).expand(Tmb, 2, D).reshape(-1, D).contiguous()  # [2*Tmb, D]
        expert_id = topi.reshape(-1).to(torch.int32).contiguous()  # [2*Tmb]
        gate_w = topw.reshape(-1).contiguous()  # [2*Tmb]
        orig_idx = torch.arange(Tmb, device=device, dtype=torch.int32).repeat_interleave(
            2
        )  # [2*Tmb]

        # Destination rank = global_expert_id // local_experts
        dest_rank = torch.div(expert_id, self.local_experts, rounding_mode="floor").to(torch.int32)
        # Pack by dest_rank so each rank chunk is contiguous
        perm = torch.argsort(dest_rank, stable=True)
        x_send = x_rep.index_select(0, perm).contiguous()
        expert_send = expert_id.index_select(0, perm).contiguous()
        orig_send = orig_idx.index_select(0, perm).contiguous()
        gate_send = gate_w.index_select(0, perm).contiguous()

        # Counts per destination rank
        send_counts = torch.bincount(dest_rank, minlength=self.ep_world).to(
            torch.int32
        )  # GPU int32

        with torch.cuda.stream(self.dispatch_stream):
            with record_function(f"dispatch_counts_{mb_tag}"):
                recv_counts = exchange_counts_async(send_counts, group=self.ep_group)
            # Ensure recv_counts is ready before converting to python list.
            # (This sync is small; for 700B you'd remove it via fixed-capacity buffers / CUDA extension.) # noqa
            self.dispatch_stream.synchronize()
            send_counts_list = send_counts.cpu().tolist()
            recv_counts_list = recv_counts.cpu().tolist()

            # Payload all_to_all (x + metadata)
            with record_function(f"dispatch_payload_{mb_tag}"):
                x_recv = all_to_all_single_async(
                    x_send, send_counts_list, recv_counts_list, group=self.ep_group
                )
                e_recv = all_to_all_single_async(
                    expert_send, send_counts_list, recv_counts_list, group=self.ep_group
                )
                o_recv = all_to_all_single_async(
                    orig_send, send_counts_list, recv_counts_list, group=self.ep_group
                )
                g_recv = all_to_all_single_async(
                    gate_send, send_counts_list, recv_counts_list, group=self.ep_group
                )

        recv_pack = RecvPack(x=x_recv, expert_id=e_recv, orig_idx=o_recv, gate_w=g_recv)
        meta = DispatchMeta(
            send_counts=send_counts_list,
            recv_counts=recv_counts_list,
            orig_idx_send_order=orig_send,
        )
        return recv_pack, meta

    def _expert_compute(self, recv_pack: RecvPack, mb_tag: str):
        """
        Runs local experts on received tokens, returns y_recv in the SAME ORDER as recv_pack.x
        (order is grouped-by-source-rank due to all_to_all layout;
        we preserve it for correct return).
        """
        # Wait for dispatch stream to finish *what has been enqueued so far*
        self.compute_stream.wait_stream(self.dispatch_stream)

        with torch.cuda.stream(self.compute_stream):
            with record_function(f"expert_compute_{mb_tag}"):
                x = recv_pack.x
                expert_id_global = recv_pack.expert_id.to(torch.int32)
                gate_w = recv_pack.gate_w

                # Map global expert id -> local expert index on this rank
                local_base = self.ep_rank * self.local_experts
                local_e = (expert_id_global - local_base).to(torch.int64)  # [Nrecv]
                # Sort by local expert for better locality (then unsort back)
                perm = torch.argsort(local_e, stable=True)
                x_s = x.index_select(0, perm)
                le_s = local_e.index_select(0, perm)

                y_s = torch.empty_like(x_s)
                # Process expert groups (simple loop; production: fused grouped GEMMs)
                start = 0
                while start < le_s.numel():
                    e = int(le_s[start].item())
                    end = start + 1
                    while end < le_s.numel() and int(le_s[end].item()) == e:
                        end += 1
                    y_s[start:end] = self.experts[e](x_s[start:end])
                    start = end

                # Unsort to original received order
                y = torch.empty_like(y_s)
                y.index_copy_(0, perm, y_s)

                # Apply gate weight on expert side (so origin just scatters + sums)
                y = y * gate_w.unsqueeze(1)

        return y

    def _combine_and_assemble(
        self, y_recv: torch.Tensor, meta: DispatchMeta, Tmb: int, mb_tag: str
    ):
        """
        Return y back to origin ranks (combine), then scatter-add into [Tmb, D] (assemble).
        """
        # Combine waits on compute
        self.combine_stream.wait_stream(self.compute_stream)

        with torch.cuda.stream(self.combine_stream):
            with record_function(f"combine_return_{mb_tag}"):
                # send split sizes = recv_counts (we are sending back to the ranks that sourced these tokens) # noqa
                # recv split sizes = send_counts (we'll receive back in the same order we sent out)
                y_back = all_to_all_single_async(
                    y_recv.contiguous(),
                    in_splits=meta.recv_counts,
                    out_splits=meta.send_counts,
                    group=self.ep_group,
                )

        # Assemble on separate stream to overlap with next compute
        self.assemble_stream.wait_stream(self.combine_stream)

        with torch.cuda.stream(self.assemble_stream):
            with record_function(f"assemble_{mb_tag}"):
                out_mb = torch.zeros(
                    (Tmb, y_back.shape[1]), device=y_back.device, dtype=y_back.dtype
                )
                # y_back is aligned with our send order, and meta.orig_idx_send_order is that same order. # noqa
                out_mb.index_add_(0, meta.orig_idx_send_order.to(torch.int64), y_back)

        return out_mb

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, D]
        returns: [B, S, D]
        """
        B, S, D = x.shape
        T = B * S
        x_flat = x.reshape(T, D)

        # 2-way microbatch split
        T0 = T // 2
        x0 = x_flat[:T0]
        x1 = x_flat[T0:]

        # ---- DeepEP-style overlap pipeline ----
        # Kick off dispatch for mb0
        recv0, meta0 = self._dispatch_one_microbatch(x0, "mb0")

        # Compute mb0 while dispatch mb1 in parallel
        y0 = self._expert_compute(recv0, "mb0")

        # Kick off dispatch for mb1 while mb0 compute is enqueued/running
        recv1, meta1 = self._dispatch_one_microbatch(x1, "mb1")

        # Start combine+assemble for mb0, overlap with mb1 compute
        out0 = self._combine_and_assemble(y0, meta0, T0, "mb0")

        # Compute mb1
        y1 = self._expert_compute(recv1, "mb1")

        # Combine+assemble mb1
        out1 = self._combine_and_assemble(y1, meta1, T - T0, "mb1")

        # Make sure assemble is done before returning
        torch.cuda.current_stream().wait_stream(self.assemble_stream)

        out = torch.cat([out0, out1], dim=0).reshape(B, S, D)
        return out


##
# Cell 5 — Tiny Transformer block using the Mo
##


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, local_experts):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = DeepEPMoE(d_model=d_model, d_hidden=d_hidden, local_experts=local_experts)

    def forward(self, x):
        with record_function("attention"):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + a
        with record_function("moe"):
            h = self.ln2(x)
            x = x + self.moe(h)
        return x


class TinyModel(nn.Module):
    def __init__(self, vocab, d_model=512, n_heads=8, d_hidden=2048, layers=2, local_experts=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(d_model, n_heads, d_hidden, local_experts) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.lm = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens):
        x = self.emb(tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.lm(x)


##
# Cell 6 — Training step + profiler export (Perfetto)
##


def run_profiled_demo(
    steps=10,
    batch=4,
    seqlen=256,
    vocab=32000,
    d_model=512,
    n_heads=8,
    d_hidden=2048,
    layers=2,
    local_experts=2,
    out_dir="./traces",
    warmup_iters=3,
):
    rank, world, local_rank = ddp_init()
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda", local_rank)
    model = TinyModel(
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        d_hidden=d_hidden,
        layers=layers,
        local_experts=local_experts,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Build TWO batches: tiny warmup batch + real batch
    torch.manual_seed(1234 + rank)

    warm_tokens = torch.randint(0, vocab, (max(1, batch // 4), max(16, seqlen // 8)), device=device)
    warm_targets = torch.randint(0, vocab, warm_tokens.shape, device=device)

    tokens = torch.randint(0, vocab, (batch, seqlen), device=device)
    targets = torch.randint(0, vocab, (batch, seqlen), device=device)

    # ---- Warmup: no profiler ----
    barrier()
    r0log(f"[warmup] starting {warmup_iters} iters | warm shape={tuple(warm_tokens.shape)}")
    for i in range(warmup_iters):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(warm_tokens)
        loss = F.cross_entropy(logits.reshape(-1, vocab), warm_targets.reshape(-1))
        loss.backward()
        opt.step()
        sync()
        dt = (time.perf_counter() - t0) * 1000
        r0log(f"[warmup] iter {i+1}/{warmup_iters} loss={loss.item():.4f} time={dt:.1f} ms")

    barrier()
    r0log("[warmup] done. starting profiled run...")

    # ---- Profiled run ----
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=4, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    prof.start()
    for step in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        with record_function("forward"):
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))

        with record_function("backward"):
            loss.backward()

        with record_function("opt_step"):
            opt.step()

        prof.step()
        sync()

        dt = (time.perf_counter() - t0) * 1000
        r0log(f"[train] step {step+1}/{steps} loss={loss.item():.4f} time={dt:.1f} ms")

    prof.stop()

    trace_path = os.path.join(out_dir, f"rank{rank}_trace.json")
    if rank == 0:
        r0log(f"[trace] exporting {trace_path}")
    prof.export_chrome_trace(trace_path)
    if rank == 0:
        r0log("[trace] export done")

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

    return trace_path


##
# Cell 7: spawn from a notebook cell
##

import torch.multiprocessing as mp  # noqa


def _worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)  # single-node
    run_profiled_demo(steps=10, batch=4, seqlen=256, layers=2, local_experts=2)


# IMPORTANT: fork avoids pickling notebook __main__
mp.start_processes(
    _worker,
    args=(2,),
    nprocs=2,
    start_method="fork",
    join=True,
)
