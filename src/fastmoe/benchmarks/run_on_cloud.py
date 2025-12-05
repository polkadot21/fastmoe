import torch
from loguru import logger

from fastmoe.config import MoEScale, get_config, init_app
from fastmoe.kernels.ops import weighted_scatter_add


def benchmark_standard(expert_outputs, indices, weights, out_shape, steps):
    """
    Standard: Cat -> Weight -> IndexAdd
    """
    # Force garbage collection to get clean memory reading
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        # 1. Cat (Allocates huge buffer)
        combined = torch.cat(expert_outputs, dim=0)
        # 2. Weight
        weighted = combined * weights.unsqueeze(-1)
        # 3. Index Add
        out = torch.zeros(out_shape, device=expert_outputs[0].device)
        out.index_add_(0, indices, weighted)

    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def benchmark_fastmoe(expert_outputs, indices, weights, out_shape, steps):
    """
    FastMoE: Loop -> Fused Kernel (In-Place)
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = expert_outputs[0].device
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        # 1. Single Allocation
        out_fused = torch.zeros(out_shape, device=device)

        offset = 0
        for expert_tensor in expert_outputs:
            k_len = expert_tensor.shape[0]

            idx_chunk = indices[offset : offset + k_len]
            w_chunk = weights[offset : offset + k_len]

            # In-Place Update (No extra buffer)
            weighted_scatter_add(expert_tensor, idx_chunk, w_chunk, out_shape, out=out_fused)
            offset += k_len

    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def run_on_cloud():
    init_app()

    # --- SELECT TARGET SCALE HERE ---
    # Change to GIGACHAT_10B or GIGACHAT_700B
    TARGET_SCALE = MoEScale.GIGACHAT_700B

    cfg = get_config(TARGET_SCALE)

    if not torch.cuda.is_available():
        logger.error("CUDA not found. Cannot run benchmark.")
        return

    device = torch.device("cuda")
    logger.info(
        f"Target: {cfg.scale.value.upper()} | D={cfg.hidden_dim}, Experts={cfg.num_experts}"
    )
    logger.info(f"Activation Volume: {cfg.activation_volume_mb:.2f} MB per pass (Float32)")

    # Data Generation
    logger.info("Allocating tensors...")
    total_tokens = cfg.total_tokens
    chunk_size = total_tokens // cfg.num_experts

    # Simulate disjoint expert outputs
    expert_outputs = [
        torch.randn(chunk_size, cfg.hidden_dim, device=device) for _ in range(cfg.num_experts)
    ]

    indices = torch.randint(0, cfg.batch_size * cfg.seq_len, (total_tokens,), device=device)
    weights = torch.rand(total_tokens, device=device)
    out_shape = (cfg.batch_size * cfg.seq_len, cfg.hidden_dim)

    # Baseline Memory (Just the inputs)
    base_mem = torch.cuda.memory_allocated() / (1024**2)
    logger.info(f"Base Memory (Inputs): {base_mem:.2f} MB")

    # Warmup
    logger.info("Warming up...")
    benchmark_standard(expert_outputs, indices, weights, out_shape, cfg.warmup_steps)
    benchmark_fastmoe(expert_outputs, indices, weights, out_shape, cfg.warmup_steps)

    # Run Benchmarks
    logger.info(f"Running Standard (Steps={cfg.active_steps})...")
    std_ms, std_mem = benchmark_standard(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )

    logger.info(f"Running FastMoE (Steps={cfg.active_steps})...")
    fast_ms, fast_mem = benchmark_fastmoe(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )

    # Report
    logger.info("-" * 40)
    logger.info(f"RESULTS | {cfg.scale.value.upper()}")
    logger.info("-" * 40)

    logger.info(f"{'Metric':<15} | {'Standard':<12} | {'FastMoE':<12} | {'Delta':<10}")
    logger.info(
        f"{'Latency (ms)':<15} | {std_ms:<12.3f} | {fast_ms:<12.3f} | {std_ms / fast_ms:.2f}x Faster"  # noqa
    )

    # Calculate Memory Overhead (Peak - Base)
    std_overhead = std_mem - base_mem
    fast_overhead = fast_mem - base_mem
    logger.info(
        f"{'Peak Mem (MB)':<15} | {std_mem:<12.0f} | {fast_mem:<12.0f} | {std_overhead / fast_overhead:.2f}x Less Ovhd"  # noqa
    )

    logger.success("Benchmark Complete.")
