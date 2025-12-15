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
        # "env://" tells PyTorch to read MASTER_ADDR, RANK, etc. from environment
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
        return 0.0  # Timing not relevant during profiling
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
        return start.elapsed_time(end) / steps


def _worker_entrypoint():
    """
    The actual benchmarking logic that runs on each GPU.
    """
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Config matching 2x H100 capacity
    cfg = get_config(MoEScale.GIGACHAT_10B)
    cfg.num_experts = 8
    local_batch_size = 4  # Divisible by 2 for pipeline

    if rank == 0:
        logger.info(f"Running Distributed Benchmark on {world_size} GPUs")
        logger.info(f"Config: {cfg.scale.value} | Batch: {local_batch_size} | Seq: {cfg.seq_len}")

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
    # 1. STANDARD IMPLEMENTATION
    # ---------------------------------------------------------
    model_std = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=16,
            ff_dim=1024,
            n_layers=2,
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
                1024,
                num_experts=cfg.num_experts,
                implementation=MoEImplementation.FAST,
                group=None,
            )
            .to(device)
            .to(torch.bfloat16)
        )

    if rank == 0:
        logger.info("Benchmarking Standard (No Overlap)...")
    # We do a quick timing run first, then profile
    ms_std = benchmark_step(model_std, x, "standard_moe_timing", profile_trace=False)
    benchmark_step(model_std, x, "standard_moe", profile_trace=True)  # Trace run

    del model_std
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. PIPELINED IMPLEMENTATION (OVERLAP)
    # ---------------------------------------------------------
    model_pipe = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=16,
            ff_dim=1024,
            n_layers=2,
            num_experts=cfg.num_experts,
            implementation=MoEImplementation.FAST,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    # Create dedicated Comm Stream
    comm_stream = torch.cuda.Stream(priority=-1)

    # Apply Patch
    for i, block in enumerate(model_pipe.blocks):
        dist_moe = (
            MoEFeedForward(
                cfg.hidden_dim,
                1024,
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
    ms_pipe = benchmark_step(model_pipe, x, "fast_moe_timing", profile_trace=False)
    benchmark_step(model_pipe, x, "fast_moe", profile_trace=True)  # Trace run

    if rank == 0:
        logger.success(f"Standard: {ms_std:.2f} ms")
        logger.success(f"Pipelined: {ms_pipe:.2f} ms")
        logger.success(f"Speedup: {ms_std / ms_pipe:.2f}x")
        logger.info("Traces saved to ./logs/standard_moe and ./logs/fast_moe")

    cleanup()


# =============================================================================
#  Launcher Logic (Notebook / Single Script Support)
# =============================================================================


def _spawn_worker(rank, world_size):
    """
    The wrapper function passed to mp.spawn.
    Sets up the environment variables needed by 'setup()'
    """
    os.environ["MASTER_ADDR"] = "localhost"
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
    """
    Public Entrypoint.
    Auto-detects whether to run as a worker or spawn a cluster.
    """
    # 1. Check if we are already inside a distributed job (e.g. torchrun)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _worker_entrypoint()
        return

    # 2. Notebook / Direct execution mode -> Spawn Processes
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning(
            f"Detected {world_size} GPU. Pipeline Parallelism requires at least 2 for effective demo."  # noqa
        )
        # Proceed anyway for debugging if user forces it

    logger.info(f"Spawning {world_size} workers for DeepSeek-V3 EP Benchmark...")

    mp.spawn(_spawn_worker, args=(world_size,), nprocs=world_size, join=True)
