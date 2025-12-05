import torch
from loguru import logger

from fastmoe.config import MoEScale, get_config, init_app
from fastmoe.kernels.ops import weighted_scatter_add


def verify_correctness(expert_outputs, indices, weights, out_shape):
    """
    Run both implementations once and compare outputs numerically.
    """
    logger.info("Verifying numerical correctness...")
    device = expert_outputs[0].device

    # 1. Run Standard Path
    combined = torch.cat(expert_outputs, dim=0)
    weighted = combined * weights.unsqueeze(-1)
    out_std = torch.zeros(out_shape, device=device)
    out_std.index_add_(0, indices, weighted)

    # 2. Run FastMoE Path (Manual Loop)
    out_fast = torch.zeros(out_shape, device=device)
    offset = 0
    for t in expert_outputs:
        k_len = t.shape[0]
        idx_chunk = indices[offset : offset + k_len]
        w_chunk = weights[offset : offset + k_len]
        weighted_scatter_add(t, idx_chunk, w_chunk, out_shape, out=out_fast)
        offset += k_len

    torch.cuda.synchronize()

    # 3. Compare
    # Calculate max absolute difference
    diff = (out_std - out_fast).abs().max()

    # Check tolerance (Float32 precision is usually around 1e-6,
    # but accumulation of thousands of floats can drift to 1e-4 or 1e-5)
    tol = 1e-4
    if diff > tol:
        logger.warning(f"Numerical Mismatch! Max Diff: {diff:.6f} (Tolerance: {tol})")
        logger.warning(
            "Note: Small differences are expected due to non-deterministic Atomic Add ordering."
        )
    else:
        logger.success(f"Verification PASSED. Max Diff: {diff:.6e}")


def benchmark_standard(expert_outputs, indices, weights, out_shape, steps):
    """Standard PyTorch: Cat (Allocation) + IndexAdd"""
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


def benchmark_fastmoe_graphed(expert_outputs, indices, weights, out_shape, steps):
    """FastMoE: CUDA Graph Replay"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = expert_outputs[0].device

    static_out = torch.zeros(out_shape, device=device)
    g = torch.cuda.CUDAGraph()

    # Warmup
    offset = 0
    for t in expert_outputs:
        k_len = t.shape[0]
        idx_chunk = indices[offset : offset + k_len]
        w_chunk = weights[offset : offset + k_len]
        weighted_scatter_add(t, idx_chunk, w_chunk, out_shape, out=static_out)
        offset += k_len
    torch.cuda.synchronize()

    # Capture
    static_out.zero_()
    with torch.cuda.graph(g):
        offset = 0
        for t in expert_outputs:
            k_len = t.shape[0]
            idx_chunk = indices[offset : offset + k_len]
            w_chunk = weights[offset : offset + k_len]
            weighted_scatter_add(t, idx_chunk, w_chunk, out_shape, out=static_out)
            offset += k_len

    # Replay
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        static_out.zero_()
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def run_on_cloud():
    init_app()
    TARGET_SCALE = MoEScale.GIGACHAT_700B
    cfg = get_config(TARGET_SCALE)

    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    logger.info(f"Target: {cfg.scale.value.upper()} | Experts: {cfg.num_experts}")

    # Setup Data
    total_tokens = cfg.total_tokens
    chunk_size = total_tokens // cfg.num_experts

    expert_outputs = [
        torch.randn(chunk_size, cfg.hidden_dim, device=device) for _ in range(cfg.num_experts)
    ]
    indices = torch.randint(0, cfg.batch_size * cfg.seq_len, (total_tokens,), device=device)
    weights = torch.rand(total_tokens, device=device)
    out_shape = (cfg.batch_size * cfg.seq_len, cfg.hidden_dim)

    base_mem = torch.cuda.memory_allocated() / (1024**2)

    # --- 1. VERIFY CORRECTNESS ---
    verify_correctness(expert_outputs, indices, weights, out_shape)

    # --- 2. BENCHMARK ---
    logger.info("Benchmarking...")

    std_ms, std_mem = benchmark_standard(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )
    fast_ms, fast_mem = benchmark_fastmoe_graphed(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )

    logger.info("=" * 60)
    logger.info(f"RESULTS | {cfg.scale.value.upper()}")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<15} | {'Standard':<12} | {'FastMoE':<12} | {'Delta':<10}")
    logger.info("-" * 60)
    logger.info(
        f"{'Latency (ms)':<15} | {std_ms:<12.3f} | {fast_ms:<12.3f} | {std_ms / fast_ms:.2f}x Faster"  # noqa
    )

    std_overhead = std_mem - base_mem
    fast_overhead = fast_mem - base_mem
    if fast_overhead < 1:
        fast_overhead = 1

    logger.info(
        f"{'Peak Mem (MB)':<15} | {std_mem:<12.0f} | {fast_mem:<12.0f} | {std_overhead / fast_overhead:.2f}x Less Ovhd"  # noqa
    )
    logger.info("=" * 60)
