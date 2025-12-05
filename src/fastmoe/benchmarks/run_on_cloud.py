import torch
from loguru import logger

from fastmoe.config import MoEScale, get_config, init_app
from fastmoe.kernels.ops import weighted_scatter_add


def benchmark_standard(expert_outputs_src, indices, weights, out_shape, steps):
    """
    Standard: Cat -> Weight -> IndexAdd
    Note: 'expert_outputs_src' is our source of fresh data.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = expert_outputs_src[0].device

    # We need to clone inputs in the loop to simulate "fresh" pointers/data
    # or just use them directly if we assume they come from previous layer.
    # To be strictly fair against the Graph version (which does a copy),
    # we should arguably just run the ops. The 'cat' allocates new memory anyway.

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        # 1. Cat (The bottleneck)
        # In a real step, these tensors come from the experts.
        # 'cat' reads them from HBM and writes to a new buffer.
        combined = torch.cat(expert_outputs_src, dim=0)

        # 2. Weight
        weighted = combined * weights.unsqueeze(-1)

        # 3. Index Add
        out = torch.zeros(out_shape, device=device)
        out.index_add_(0, indices, weighted)

    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def benchmark_fastmoe_graphed(expert_outputs_src, indices, weights, out_shape, steps):
    """
    FastMoE with CUDA Graphs + Realistic Data Flow.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = expert_outputs_src[0].device

    # 1. Allocate STATIC Staging Buffers (The "Graph Memory")
    # In a real static graph training (like with torch.compile),
    # the model layers have fixed output buffers.
    static_expert_outputs = [torch.zeros_like(t) for t in expert_outputs_src]
    static_out = torch.zeros(out_shape, device=device)

    # 2. Capture Graph on the STATIC buffers
    g = torch.cuda.CUDAGraph()

    # Warmup for capture
    offset = 0
    for t in static_expert_outputs:
        k_len = t.shape[0]
        # Use simple slicing for metadata (assumed static for graph capture,
        # or we'd need to copy metadata too. In MoE, indices CHANGE every step.
        # Capturing Dynamic MoE with CUDA Graphs is tricky!
        # We usually capture the "Expert Computation" but not the routing if sizes change.
        #
        # CRITICAL REALISM CHECK:
        # If sizes change every step (standard MoE), you CANNOT use CUDA Graphs easily.
        # You must use 'torch.compile' with dynamic shapes or eager mode.
        #
        # FOR THIS BENCHMARK:
        # We assume a fixed-size or padded regime (like MegaBlocks or padded MoE)
        # where tensor sizes are stable, allowing Graphs.
        idx_chunk = indices[offset : offset + k_len]
        w_chunk = weights[offset : offset + k_len]
        weighted_scatter_add(t, idx_chunk, w_chunk, out_shape, out=static_out)
        offset += k_len
    torch.cuda.synchronize()

    # Capture
    static_out.zero_()
    with torch.cuda.graph(g):
        offset = 0
        for t in static_expert_outputs:
            k_len = t.shape[0]
            idx_chunk = indices[offset : offset + k_len]
            w_chunk = weights[offset : offset + k_len]

            # The kernel records using the pointers of 'static_expert_outputs'
            weighted_scatter_add(t, idx_chunk, w_chunk, out_shape, out=static_out)
            offset += k_len

    # 3. Run Benchmark with Data Refresh
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Pre-generate "fresh" data batches to avoid CPU generation cost inside timing
    # (Simulating data arriving from previous GPU kernel)
    fresh_data_batches = []
    for _ in range(5):  # Cycle through 5 batches
        fresh_data_batches.append([torch.randn_like(t) for t in expert_outputs_src])

    start_event.record()
    for i in range(steps):
        # A. Simulate "Previous Layer" Output
        # We copy fresh random data into the static buffers that the Graph knows.
        # This forces HBM writes (cache flush).
        current_batch = fresh_data_batches[i % 5]
        for src, dst in zip(current_batch, static_expert_outputs):  # noqa
            dst.copy_(src)  # Device-to-Device copy (Fast, but non-zero cost)

        # B. Reset Output
        static_out.zero_()

        # C. Replay Graph (Compute)
        g.replay()

    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    return start_event.elapsed_time(end_event) / steps, peak_mem


def run_on_cloud():
    init_app()

    # --- ULTRA SCALE ---
    TARGET_SCALE = MoEScale.GIGACHAT_ULTRA_700B

    cfg = get_config(TARGET_SCALE)
    if not torch.cuda.is_available():
        logger.error("CUDA required.")
        return

    device = torch.device("cuda")
    logger.info(f"Target: {cfg.scale.value.upper()}")
    logger.info(f"Experts: {cfg.num_experts} | Hidden: {cfg.hidden_dim}")

    # Initial Data Allocations
    total_tokens = cfg.total_tokens
    chunk_size = total_tokens // cfg.num_experts

    expert_outputs = [
        torch.randn(chunk_size, cfg.hidden_dim, device=device) for _ in range(cfg.num_experts)
    ]

    indices = torch.randint(0, cfg.batch_size * cfg.seq_len, (total_tokens,), device=device)
    weights = torch.rand(total_tokens, device=device)
    out_shape = (cfg.batch_size * cfg.seq_len, cfg.hidden_dim)

    base_mem = torch.cuda.memory_allocated() / (1024**2)
    logger.info(f"Base Memory: {base_mem:.2f} MB")

    # Run Standard
    logger.info(f"Benchmarking Standard (Steps={cfg.active_steps})...")
    std_ms, std_mem = benchmark_standard(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )

    # Run FastMoE (Graphed + Data Copy)
    logger.info(f"Benchmarking FastMoE + Graphs + FreshData (Steps={cfg.active_steps})...")
    fast_ms, fast_mem = benchmark_fastmoe_graphed(
        expert_outputs, indices, weights, out_shape, cfg.active_steps
    )

    # Report
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
    logger.info(
        f"{'Peak Mem (MB)':<15} | {std_mem:<12.0f} | {fast_mem:<12.0f} | {std_overhead / fast_overhead:.2f}x Less Ovhd"  # noqa
    )
    logger.info("=" * 60)
