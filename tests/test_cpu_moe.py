from contextlib import contextmanager  # <--- Import this

import torch

# --- PATCHING (Mock CUDA) ---
from mocks import MockEvent, MockStream

# 1. Patch Classes
torch.cuda.Event = MockEvent
torch.cuda.Stream = MockStream

# 2. Patch Functions
torch.cuda.current_stream = lambda: MockStream()
torch.cuda.is_available = lambda: True
torch.cuda.synchronize = lambda: None


# 3. Patch Context Manager (The Fix)
@contextmanager
def mock_cuda_stream_context(stream):
    yield


torch.cuda.stream = mock_cuda_stream_context
# ----------------------------
from fastmoe.kernels.ops import weighted_scatter_add  # noqa
from fastmoe.layers.moe import MoEFeedForward  # noqa
from fastmoe.layers.pipelined_block import PipelinedMoEBlock  # noqa
from fastmoe.models.tiny_model import TinyModel  # noqa


def test_moe_forward_backward_cpu():
    """Verifies MoE gradient flow (Simulated on CPU/Mock CUDA)."""
    torch.manual_seed(42)
    B, T, D = 2, 8, 32
    ff_dim = 64
    num_experts = 4
    top_k = 2

    moe = MoEFeedForward(D, ff_dim, num_experts=num_experts, top_k=top_k)
    x = torch.randn(B, T, D, requires_grad=True)
    target = torch.randn(B, T, D)

    # Forward
    out = moe(x)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()

    # Backward
    loss = ((out - target) ** 2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Check experts got gradients
    active_experts = 0
    for _, expert in enumerate(moe.experts):
        if expert[0].weight.grad is not None:
            active_experts += 1
    assert active_experts > 0, "No experts received gradients!"


def test_pipelined_block_logic():
    """
    Verifies that PipelinedMoEBlock correctly splits and reassembles
    the batch, producing the same result as a sequential run.
    """
    torch.manual_seed(42)
    D = 32

    # Simple Mock Layers
    attn = torch.nn.Linear(D, D)  # Fake Attention
    moe = torch.nn.Linear(D, D)  # Fake MoE (Compute only for test)

    block = PipelinedMoEBlock(attn, moe, hidden_dim=D)

    # Ensure streams are active in the test env
    block.use_streams = True

    # Batch size must be >= 2 for pipelining to trigger
    B, T = 4, 10
    x = torch.randn(B, T, D)

    # 1. Run Pipeline Forward
    out_pipe = block(x)

    # 2. Run Sequential Forward (Ground Truth)
    # We manually force the sequential path or replicate logic
    out_seq = block._forward_sequential(x)

    # The pipeline splits execution order but math should be identical
    # (ignoring float precision noise if real GPU)
    assert torch.allclose(out_pipe, out_seq, atol=1e-6), "Pipeline output differs from Sequential!"
    print("\n[Test] Pipelined Block Logic Passed.")


def test_full_model_integration():
    """Verifies TinyModel with FastMoE implementation."""
    print("\n--- [Test] Full Model Integration ---")
    in_dim, dim = 16, 32
    # 'FAST' triggers PipelinedMoEBlock inside TinyModel
    model = TinyModel(
        in_dim=in_dim,
        dim=dim,
        n_heads=4,
        ff_dim=64,
        n_layers=2,
        num_experts=4,
        implementation="fast",
    )

    x = torch.randn(4, 10, in_dim)  # Batch 4 for pipelining
    out = model(x)

    assert out.shape == (4, 10, in_dim)
    print("Full model forward pass successful.")


def test_scatter_add_correctness():
    """Numerically verify the Scatter-Add CPU fallback."""
    print("\n--- [Test] Kernel Correctness (CPU) ---")
    N, D = 10, 4
    src = torch.randn(N, D)
    weights = torch.rand(N)
    indices = torch.randint(0, 5, (N,))
    out_shape = (5, D)

    # FastMoE Fallback
    out_fast = weighted_scatter_add(src, indices, weights, out_shape)

    # Naive Loop
    out_gt = torch.zeros(out_shape)
    for i in range(N):
        dest_idx = indices[i].item()
        out_gt[dest_idx] += src[i] * weights[i]

    diff = (out_fast - out_gt).abs().max()
    assert diff < 1e-5
    print("Numerical verification passed.")


if __name__ == "__main__":
    test_scatter_add_correctness()
    test_moe_forward_backward_cpu()
    test_pipelined_block_logic()
    test_full_model_integration()
