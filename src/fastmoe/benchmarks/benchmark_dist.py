import torch
import torch.distributed as dist
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

from fastmoe.config import MoEScale, get_config
from fastmoe.consts import MoEImplementation
from fastmoe.layers.moe import MoEFeedForward
from fastmoe.layers.pipeline import PipelinedMoEBlock
from fastmoe.models.tiny_model import TinyModel


def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())


def cleanup():
    dist.destroy_process_group()


def benchmark_step(model, x, desc, profile_trace=False):
    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()

    steps = 10

    if profile_trace:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./logs/{desc}"),
        ) as prof:
            for _ in range(steps):
                with record_function("Model Step"):
                    out = model(x)
                    out.mean().backward()
                prof.step()
    else:
        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            out = model(x)
            out.mean().backward()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / steps


def run_benchmark():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = get_config(MoEScale.GIGACHAT_10B)  # Use manageable size

    # Adjust config for distributed
    cfg.num_experts = 8  # Total experts
    local_batch_size = 4  # Must be divisible by 2 for pipeline

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

    # ==========================================
    # 1. STANDARD IMPLEMENTATION
    # ==========================================
    model_std = (
        TinyModel(
            in_dim=cfg.hidden_dim,
            dim=cfg.hidden_dim,
            n_heads=16,
            ff_dim=1024,
            n_layers=2,
            num_experts=cfg.num_experts,
            implementation=MoEImplementation.FAST,  # Use Fast Kernel, but Standard Pipeline
        )
        .to(device)
        .to(torch.bfloat16)
    )

    # Patch layers with Distributed MoE
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

    # Run Standard
    if rank == 0:
        logger.info("Benchmarking Standard (No Overlap)...")
    ms_std = benchmark_step(model_std, x, "standard_moe", profile_trace=True)

    del model_std
    torch.cuda.empty_cache()

    # ==========================================
    # 2. PIPELINED IMPLEMENTATION (OVERLAP)
    # ==========================================
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
    comm_stream = torch.cuda.Stream(priority=-1)  # High priority? Or standard.

    # Apply Patch
    for i, block in enumerate(model_pipe.blocks):
        # 1. Create Dist MoE
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

        # 2. Replace FF
        block.ff = dist_moe

        # 3. Wrap Block in Pipeline
        model_pipe.blocks[i] = PipelinedMoEBlock(block, comm_stream)

    if rank == 0:
        logger.info("Benchmarking Fast (Pipelined Overlap)...")
    ms_pipe = benchmark_step(model_pipe, x, "fast_moe", profile_trace=True)

    if rank == 0:
        logger.success(f"Standard: {ms_std:.2f} ms")
        logger.success(f"Pipelined: {ms_pipe:.2f} ms")
        logger.success(f"Speedup: {ms_std / ms_pipe:.2f}x")
        logger.info("Traces saved to ./logs/standard_moe and ./logs/fast_moe")

    cleanup()
