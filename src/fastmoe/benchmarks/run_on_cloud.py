import gc
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function, schedule

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
# 2. Training Loop (Realistic Load + Profiling)
# -------------------------------------------------------------------------


def run_training_experiment(
    implementation: str, cfg: MoESetup, rank: int, trace_filename: str = None
):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(f"cuda:{rank}")

    # Use a heavier configuration for the training loop to emphasize compute
    model = TinyModel(
        in_dim=cfg.hidden_dim,
        dim=cfg.hidden_dim,
        n_heads=32,  # Heavy attention
        ff_dim=cfg.hidden_dim * 4,  # Standard MLP ratio (4x)
        n_layers=2,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        implementation=implementation,
        use_moe=True,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    B, T, D = cfg.batch_size, cfg.seq_len, cfg.hidden_dim
    x = torch.randn(B, T, D, device=device)
    target = x.clone()

    if rank == 0:
        logger.info(f"Running {implementation.upper()} Training Loop...")
        if trace_filename:
            logger.info(f"Profiling enabled. Saving trace to ./traces/{trace_filename}")

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

    # --- Profiling Setup ---
    prof = None
    if trace_filename:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            with_stack=True,
            on_trace_ready=None,  # We export manually
        )
        prof.start()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()

    steps = 10 if trace_filename else cfg.active_steps

    start_event.record()
    for step in range(steps):
        # Giant Scope to measure the ENTIRE step cost (Python + GPU)
        with record_function(f"Global Step {step} [{implementation.upper()}]"):
            with record_function("Forward"):
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    y = model(x)
                    loss = ((y - target) ** 2).mean()

            with record_function("Backward"):
                scaler.scale(loss).backward()

            with record_function("Optimizer"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if rank == 0 and not trace_filename and (step % 10 == 0 or step == steps - 1):
            curr_mem = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Step {step:03d} | Loss: {loss.item():.4f} | Mem: {curr_mem:.2f} GB")

        if prof:
            prof.step()

    end_event.record()
    torch.cuda.synchronize()

    if prof:
        prof.stop()
        if rank == 0:
            prof.export_chrome_trace(f"./traces/{trace_filename}")
            logger.success(f"Trace saved: ./traces/{trace_filename}")

    avg_time = start_event.elapsed_time(end_event) / steps
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    return avg_time, peak_mem


# -------------------------------------------------------------------------
# 3. Main Worker (Called by either torchrun or mp.spawn)
# -------------------------------------------------------------------------


def main_worker(rank, world_size):
    """
    The actual entry point for the process.
    rank: Global rank of the process (0..N-1).
    world_size: Total number of GPUs.
    """
    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    TARGET_SCALE = MoEScale.TRACE_OPTIMIZED
    cfg = get_config(TARGET_SCALE)

    if rank == 0:
        logger.info(f"Target: {cfg.scale.value.upper()} | Experts: {cfg.num_experts}")

    # --- PART 1: KERNEL BENCHMARK ---
    device = torch.device(f"cuda:{rank}")

    total_tokens = cfg.total_tokens
    chunk_size = total_tokens // cfg.num_experts

    expert_outputs = [
        torch.randn(chunk_size, cfg.hidden_dim, device=device) for _ in range(cfg.num_experts)
    ]
    indices = torch.randint(0, cfg.batch_size * cfg.seq_len, (total_tokens,), device=device)
    weights = torch.rand(total_tokens, device=device)
    out_shape = (cfg.batch_size * cfg.seq_len, cfg.hidden_dim)

    base_mem = torch.cuda.memory_allocated() / (1024**2)

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

    del expert_outputs, indices, weights
    torch.cuda.empty_cache()

    # --- PART 2: TRAINING BENCHMARK + TRACING ---
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"TRAINING BENCHMARK & PROFILING | {TARGET_SCALE.value.upper()}")

    # 1. Profile STANDARD
    try:
        std_time, std_mem = run_training_experiment(
            consts.MoEImplementation.STANDARD, cfg, rank, trace_filename="trace_standard.json"
        )
        if rank == 0:
            logger.info(f"Standard: {std_time:.2f} ms/step | Peak Mem: {std_mem:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            logger.error("Standard: OOM!")
        std_time, std_mem = float("inf"), float("inf")

    # 2. Profile FAST
    fast_time, fast_mem = run_training_experiment(
        consts.MoEImplementation.FAST, cfg, rank, trace_filename="trace_fast.json"
    )

    if rank == 0:
        logger.info(f"FastMoE:  {fast_time:.2f} ms/step | Peak Mem: {fast_mem:.2f} GB")
        if std_time != float("inf"):
            logger.success(f"Final Speedup: {std_time / fast_time:.2f}x")
            logger.success(f"Memory Saved: {std_mem - fast_mem:.2f} GB")

    dist.destroy_process_group()


# -------------------------------------------------------------------------
# 4. Entry Point
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
