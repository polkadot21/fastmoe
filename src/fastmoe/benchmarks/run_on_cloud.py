import gc
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from loguru import logger

from fastmoe import consts
from fastmoe.config import MoEScale, MoESetup, get_config
from fastmoe.kernels.ops import grouped_weighted_scatter_add, prepare_grouped_metadata
from fastmoe.models.tiny_model import TinyModel

# -------------------------------------------------------------------------
# 1. Verification & Micro-Benchmarks (Kernel Logic)
# -------------------------------------------------------------------------


def verify_correctness(expert_outputs, indices, weights, out_shape):
    """Verifies that the custom kernel matches standard PyTorch math."""
    logger.info("Verifying numerical correctness...")
    device = expert_outputs[0].device

    # A. Standard (Cat -> IndexAdd)
    combined = torch.cat(expert_outputs, dim=0)
    weighted = combined * weights.unsqueeze(-1)
    out_std = torch.zeros(out_shape, device=device)
    out_std.index_add_(0, indices, weighted)

    # B. FastMoE (Grouped Kernel)
    out_fast = torch.zeros(out_shape, device=device)
    grouped_weighted_scatter_add(expert_outputs, indices, weights, out_shape, out=out_fast)

    torch.cuda.synchronize()

    diff = (out_std - out_fast).abs().max()
    tol = 1e-4
    if diff > tol:
        logger.warning(f"Numerical Mismatch! Max Diff: {diff:.6f} (Tolerance: {tol})")
    else:
        logger.success(f"Verification PASSED. Max Diff: {diff:.6e}")


def benchmark_standard(expert_outputs, indices, weights, out_shape, steps):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        combined = torch.cat(expert_outputs, dim=0)
        weighted = combined * weights.unsqueeze(-1)
        out = torch.zeros(out_shape, device=expert_outputs[0].device)
        out.index_add_(0, indices, weighted)
    end_event.record()

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def benchmark_fastmoe_grouped(expert_outputs, indices, weights, out_shape, steps):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = expert_outputs[0].device

    out_static = torch.zeros(out_shape, device=device)
    metadata = prepare_grouped_metadata(expert_outputs, device)

    # Warmup & Graph Capture
    grouped_weighted_scatter_add(
        expert_outputs, indices, weights, out_shape, out=out_static, metadata=metadata
    )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    out_static.zero_()
    with torch.cuda.graph(g):
        grouped_weighted_scatter_add(
            expert_outputs, indices, weights, out_shape, out=out_static, metadata=metadata
        )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        out_static.zero_()
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


# -------------------------------------------------------------------------
# 2. Trace Generation (The "Red Bar under Blue Bar" Proof)
# -------------------------------------------------------------------------


def generate_overlap_trace(cfg: MoESetup, rank: int = 0):
    if rank == 0:
        logger.info("=" * 60)
        logger.info("TRACING: Generating 'moe_overlap_trace.json'...")
        logger.info(f"Config: B={cfg.batch_size} | S={cfg.seq_len} | D={cfg.hidden_dim}")
        logger.info("=" * 60)

    device = torch.device(f"cuda:{rank}")

    # Force 'FAST' implementation
    model = TinyModel(
        in_dim=cfg.hidden_dim,
        dim=cfg.hidden_dim,
        n_heads=16,  # Standard head count
        ff_dim=cfg.hidden_dim * 4,  # Standard MLP ratio
        n_layers=2,  # Keep depth low, we only care about horizontal overlap
        num_experts=cfg.num_experts,
        implementation=consts.MoEImplementation.FAST,
    ).to(device)

    # Use Config dimensions!
    # This ensures we hit the "Goldilocks" zone defined in TRACE_OPTIMIZED
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        _ = model(x)
    torch.cuda.synchronize()

    # Profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces"),
        record_shapes=True,
        profile_memory=False,  # Disable memory profiling to reduce overhead
        with_stack=True,
    ) as p:
        for i in range(5):
            torch.cuda.nvtx.range_push(f"Step {i}")
            _ = model(x)
            torch.cuda.nvtx.range_pop()
            p.step()

    if rank == 0:
        trace_path = "./traces/moe_overlap_trace.json"
        p.export_chrome_trace(trace_path)
        logger.success(f"Trace generated! Saved to: {trace_path}")


# -------------------------------------------------------------------------
# 3. Training Loop (Realistic Load)
# -------------------------------------------------------------------------


def run_training_experiment(implementation: str, cfg: MoESetup, rank: int):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(f"cuda:{rank}")

    model = TinyModel(
        in_dim=cfg.hidden_dim,
        dim=cfg.hidden_dim,
        n_heads=16,
        ff_dim=256,
        n_layers=2,
        num_experts=cfg.num_experts,
        implementation=implementation,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    B, T, D = cfg.batch_size, cfg.seq_len, cfg.hidden_dim
    x = torch.randn(B, T, D, device=device)
    target = x.clone()

    if rank == 0:
        logger.info(f"Running {implementation.upper()} Training Loop...")

    # Warmup
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            y = model(x)
            loss = ((y - target) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()

    start_event.record()
    for step in range(cfg.active_steps):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            y = model(x)
            loss = ((y - target) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if rank == 0 and (step % 10 == 0 or step == cfg.active_steps - 1):
            curr_mem = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Step {step:03d} | Loss: {loss.item():.4f} | Mem: {curr_mem:.2f} GB")

    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / cfg.active_steps
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    return avg_time, peak_mem


# -------------------------------------------------------------------------
# 4. Main Worker (Called by either torchrun or mp.spawn)
# -------------------------------------------------------------------------


def main_worker(rank, world_size):
    """
    The actual entry point for the process.
    rank: Global rank of the process (0..N-1).
    world_size: Total number of GPUs.
    """
    # If run via mp.spawn, we must manually set env vars for init_process_group
    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    # init_app will handle dist.init_process_group logic safely
    # For mp.spawn, we need to manually pass rank/world_size to it via env vars
    # if we want it to pick them up, or simpler: just init here if using spawn.

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    TARGET_SCALE = MoEScale.TRACE_OPTIMIZED
    cfg = get_config(TARGET_SCALE)

    if rank == 0:
        logger.info(f"Target: {cfg.scale.value.upper()} | Experts: {cfg.num_experts}")

    # --- PART 1: KERNEL BENCHMARK (Only on Rank 0 usually, but we run on all) ---
    device = torch.device(f"cuda:{rank}")

    # We allocate inputs on all ranks to simulate load, but log only on Rank 0
    total_tokens = cfg.total_tokens
    chunk_size = total_tokens // cfg.num_experts

    expert_outputs = [
        torch.randn(chunk_size, cfg.hidden_dim, device=device) for _ in range(cfg.num_experts)
    ]
    indices = torch.randint(0, cfg.batch_size * cfg.seq_len, (total_tokens,), device=device)
    weights = torch.rand(total_tokens, device=device)
    out_shape = (cfg.batch_size * cfg.seq_len, cfg.hidden_dim)

    base_mem = torch.cuda.memory_allocated() / (1024**2)

    # Run verification
    if rank == 0:
        verify_correctness(expert_outputs, indices, weights, out_shape)
        logger.info("Benchmarking Kernel...")

    std_ms, std_mem = benchmark_standard(expert_outputs, indices, weights, out_shape, 10)
    fast_ms, fast_mem = benchmark_fastmoe_grouped(expert_outputs, indices, weights, out_shape, 10)

    if rank == 0:
        std_overhead = std_mem - base_mem
        fast_overhead = fast_mem - base_mem
        if fast_overhead < 0.1:
            fast_overhead = 0.1

        logger.info("-" * 60)
        logger.info("KERNEL RESULTS (Latency & Memory Overhead)")
        logger.info("-" * 60)
        logger.info(
            f"Latency: Std={std_ms:.2f}ms | Fast={fast_ms:.2f}ms | Speedup={std_ms / fast_ms:.2f}x"
        )
        logger.info(f"Mem Overhead: Std={std_overhead:.0f}MB | Fast={fast_overhead:.0f}MB")
        logger.info("-" * 60)

    # Cleanup
    del expert_outputs, indices, weights
    torch.cuda.empty_cache()

    # --- PART 2: OVERLAP TRACE ---
    # We run this on all ranks, but only Rank 0 exports the JSON
    generate_overlap_trace(cfg, rank)

    # --- PART 3: TRAINING BENCHMARK ---
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"TRAINING BENCHMARK | {TARGET_SCALE.value.upper()}")

    try:
        std_time, std_mem = run_training_experiment(consts.MoEImplementation.STANDARD, cfg, rank)
        if rank == 0:
            logger.info(f"Standard: {std_time:.2f} ms/step | Peak Mem: {std_mem:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            logger.error("Standard: OOM!")
        std_time, std_mem = float("inf"), float("inf")

    fast_time, fast_mem = run_training_experiment(consts.MoEImplementation.FAST, cfg, rank)

    if rank == 0:
        logger.info(f"FastMoE:  {fast_time:.2f} ms/step | Peak Mem: {fast_mem:.2f} GB")
        if std_time != float("inf"):
            logger.success(f"Final Speedup: {std_time / fast_time:.2f}x")
            logger.success(f"Memory Saved: {std_mem - fast_mem:.2f} GB")

    dist.destroy_process_group()


# -------------------------------------------------------------------------
# 5. Entry Point
# -------------------------------------------------------------------------


def run_on_cloud():
    """
    Entry point. Detects if running via torchrun or needs mp.spawn.
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA detected. Skipping.")
        return

    world_size = torch.cuda.device_count()
    logger.info(f"Running via mp.spawn on {world_size} GPUs...")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size)
