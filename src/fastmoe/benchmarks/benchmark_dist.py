import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from fastmoe.config import MoEScale, get_config
from fastmoe.consts import MoEImplementation
from fastmoe.layers.moe import MoEFeedForward
from fastmoe.layers.pipeline import PipelinedMoEBlock
from fastmoe.models.tiny_model import TinyModel

# =============================================================================
#  Distributed Helpers
# =============================================================================


def setup():
    """Initializes the distributed process group."""
    if not dist.is_initialized():
        # [Fix] Use 127.0.0.1 explicitly to avoid IPv6/localhost ambiguity issues
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        dist.init_process_group("nccl", init_method="env://")

    # Ensure we pin the correct device for this process
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    torch.cuda.set_device(local_rank)


def cleanup():
    """Destroys the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
#  Benchmarking Logic (The Worker)
# =============================================================================


def benchmark_step(model, x, desc, profile_trace=False):
    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()

    # Reset Memory Stats for clean reading
    torch.cuda.reset_peak_memory_stats()

    steps = 10

    if profile_trace:
        # Generate Chrome Traces
        log_path = f"./logs/{desc}"
        os.makedirs(log_path, exist_ok=True)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
        ) as prof:
            for _ in range(steps):
                with record_function("Model Step"):
                    out = model(x)
                    out.mean().backward()
                prof.step()

        # Calculate Memory
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        return 0.0, peak_mem
    else:
        # Strict Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            out = model(x)
            out.mean().backward()
        end.record()
        torch.cuda.synchronize()

        avg_time = start.elapsed_time(end) / steps
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        return avg_time, peak_mem


def _worker_entrypoint():
    """
    The actual benchmarking logic that runs on each GPU.
    """
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # [Fix] Override Config to create a Communication-Heavy Scenario
    # Previous config (seq=4096, batch=4) was Compute Bound (Attention O(N^2)).
    # We switch to DeepSeek-V3 style: Wide Hidden Dim, High Batch, Low Seq.
    cfg = get_config(MoEScale.GIGACHAT_10B)

    cfg.hidden_dim = 16384  # Massive width (Increases Comm Payload)
    cfg.seq_len = 128  # Shorter seq (Reduces Attention Compute to reveal Comm cost)
    local_batch_size = 32  # Higher batch (Increases Comm Payload)
    cfg.num_experts = 8

    if rank == 0:
        logger.info(f"Running Distributed Benchmark on {world_size} GPUs")
        logger.info(
            f"High-Bandwidth Config: Hidden={cfg.hidden_dim} | Batch={local_batch_size} | Seq={cfg.seq_len}"  # noqa
        )

    device = torch.device(f"cuda:{rank}")

    # Data
    x = torch.randn(
        local_batch_size,
        cfg.seq_len,
        cfg.hidden_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # ---------------------------------------------------------
    # 1. STANDARD IMPLEMENTATION (Sequential EP)
    # ---------------------------------------------------------
    model_std = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=32,
            ff_dim=4096,
            n_layers=2,  # n_heads increased for 16k dim
            num_experts=cfg.num_experts,
            implementation=MoEImplementation.FAST,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    # Patch layers with Distributed MoE (Sequential)
    for block in model_std.blocks:
        block.ff = (
            MoEFeedForward(
                cfg.hidden_dim,
                4096,
                num_experts=cfg.num_experts,
                implementation=MoEImplementation.FAST,
                group=None,
            )
            .to(device)
            .to(torch.bfloat16)
        )

    if rank == 0:
        logger.info("Benchmarking Standard (No Overlap)...")
    ms_std, mem_std = benchmark_step(model_std, x, "standard_moe_timing", profile_trace=False)
    benchmark_step(model_std, x, "standard_moe", profile_trace=True)

    del model_std
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. PIPELINED IMPLEMENTATION (Overlap)
    # ---------------------------------------------------------
    model_pipe = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=32,
            ff_dim=4096,
            n_layers=2,
            num_experts=cfg.num_experts,
            implementation=MoEImplementation.FAST,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    # Create dedicated Comm Stream (High Priority to ensure it preempts if needed)
    comm_stream = torch.cuda.Stream(priority=-1)

    # Apply Patch
    for i, block in enumerate(model_pipe.blocks):
        dist_moe = (
            MoEFeedForward(
                cfg.hidden_dim,
                4096,
                num_experts=cfg.num_experts,
                implementation=MoEImplementation.FAST,
                group=None,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        block.ff = dist_moe
        # Wrap Block in Pipeline
        model_pipe.blocks[i] = PipelinedMoEBlock(block, comm_stream)

    if rank == 0:
        logger.info("Benchmarking Fast (Pipelined Overlap)...")
    ms_pipe, mem_pipe = benchmark_step(model_pipe, x, "fast_moe_timing", profile_trace=False)
    benchmark_step(model_pipe, x, "fast_moe", profile_trace=True)

    if rank == 0:
        logger.info("=" * 60)
        logger.info(" BENCHMARK RESULTS ")
        logger.info("=" * 60)
        logger.info(f"{'Metric':<15} | {'Standard':<15} | {'Pipelined':<15} | {'Improvement':<15}")
        logger.info("-" * 65)
        logger.info(
            f"{'Latency (ms)':<15} | {ms_std:<15.2f} | {ms_pipe:<15.2f} | {ms_std / ms_pipe:.2f}x Faster"  # noqa
        )
        logger.info(
            f"{'Memory (MB)':<15} | {mem_std:<15.0f} | {mem_pipe:<15.0f} | {mem_std - mem_pipe:.0f} MB Saved"  # noqa
        )
        logger.info("=" * 60)
        logger.info("Traces saved to ./logs/standard_moe and ./logs/fast_moe")

    cleanup()


# =============================================================================
#  Launcher Logic
# =============================================================================


def _spawn_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # [Fix] Hardcoded to avoid socket errors
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    try:
        _worker_entrypoint()
    except Exception as e:
        logger.error(f"Rank {rank} failed: {e}")
        raise e


def run_benchmark():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _worker_entrypoint()
        return

    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning(
            f"Detected {world_size} GPU. Pipeline Parallelism requires at least 2 for effective demo."  # noqa
        )

    logger.info(f"Spawning {world_size} workers for DeepSeek-V3 EP Benchmark...")

    mp.spawn(_spawn_worker, args=(world_size,), nprocs=world_size, join=True)
