import torch

# =========================================================================
#  Environment Shim (Mac/CPU vs Linux/GPU)
# =========================================================================
# Triton is an Nvidia-only compiler. To allow development and unit testing
# on Mac (Apple Silicon) or standard CPUs, we must mock the imports.
# If we detect we are not on a supported platform, we engage a CPU fallback.
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    # Mock Triton symbols to prevent NameErrors during static analysis
    from unittest.mock import MagicMock

    triton = MagicMock()
    tl = MagicMock()
    # Mock the JIT decorator to strictly return the function (so Python can read it)
    triton.jit = lambda f: f
    # Mock compile-time constants types
    tl.constexpr = int


# =========================================================================
#  Triton Kernel: The "No-Cat" Engine
# =========================================================================
# This kernel is the heart of the optimization.
# Standard MoE implementations suffer from a massive memory bandwidth bottleneck:
#   1. Experts output a List[Tensor].
#   2. torch.cat() creates a massive contiguous buffer (Read/Write penalty).
#   3. torch.index_add_() reads that massive buffer again to sort tokens.
#
# This kernel fuses those steps. It takes a pointer to a single expert's output
# and writes it DIRECTLY to its final destination in the output tensor.
# By calling this kernel in a loop over experts, we achieve:
#   Input List -> [KERNEL] -> Output Tensor
# We completely eliminate the intermediate 'concatenated' buffer allocation.
@triton.jit
def _weighted_scatter_add_kernel(
    src_ptr,  # [N, D]  - Pointer to the specific expert's output block
    idx_ptr,  # [N]     - Restoration indices (where these tokens belong in final batch)
    w_ptr,  # [N]     - The gating weights (softmax score for this expert-token pair)
    out_ptr,  # [Total, D] - Pointer to the PRE-ALLOCATED final output tensor
    stride_src_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,  # Optimization: Number of elements to process per thread block
    HIDDEN_DIM: tl.constexpr,  # The hidden dimension size
):
    # Each program instance (PID) handles one ROW (one token) from the source.
    pid = tl.program_id(0)

    # 1. Load Routing Metadata
    # We need to know: "Where does this specific token go?" and "How much does it count?"
    # These loads are scalar, so they are very cheap relative to the vector loads below.
    target_row_idx = tl.load(idx_ptr + pid)
    weight = tl.load(w_ptr + pid)

    # 2. Calculate Memory Addresses
    # Triton uses pointer arithmetic. We find the starting memory address for:
    # - The source row (Expert output)
    # - The destination row (Final buffer)
    src_row_start_ptr = src_ptr + pid * stride_src_row
    out_row_start_ptr = out_ptr + target_row_idx * stride_out_row

    # 3. Vectorized Load-Compute-Store Loop
    # We cannot load the entire Hidden Dimension (e.g., 4096) at once.
    # We break it into blocks (BLOCK_SIZE, e.g., 1024) to fit in SRAM.
    for off in range(0, HIDDEN_DIM, BLOCK_SIZE):
        # Create a range of offsets [0, 1, ... BLOCK_SIZE-1]
        cols = off + tl.arange(0, BLOCK_SIZE)

        # Boundary Check: Ensure we don't read past the end of the hidden dimension
        mask = cols < HIDDEN_DIM

        # A. Load: Fetch a vector of floats from Global Memory to SRAM
        val = tl.load(src_row_start_ptr + cols, mask=mask)

        # B. Compute: Apply the gating weight (Fusion Step 1)
        # This multiplication happens in registers, saving a global read/write
        # compared to doing 'expert_output * weight' in PyTorch.
        weighted_val = val * weight

        # C. Store: Atomic Add (Fusion Step 2)
        # Why Atomic? Because multiple experts might route to the same destination token
        # (Top-K > 1). If two blocks try to write to 'out_ptr' simultaneously,
        # we get a race condition. 'atomic_add' serializes these writes at the hardware level.
        tl.atomic_add(out_row_start_ptr + cols, weighted_val, mask=mask)


# =========================================================================
#  Autograd Function: The PyTorch Bridge
# =========================================================================
class WeightedScatterAdd(torch.autograd.Function):
    """
    This custom Autograd function wraps the Triton kernel.
    It handles the Forward pass (invoking the kernel) and the Backward pass
    (computing gradients for inputs and weights).
    """

    @staticmethod
    def forward(ctx, src, indices, weights, out_shape, out_tensor=None):
        # 1. Save for Backward
        # We need the inputs later to calculate gradients.
        ctx.save_for_backward(indices, weights, src)

        # 2. Output Buffer Management
        # The key to "No-Cat" is allowing the user to provide an existing buffer ('out_tensor').
        # If 'out_tensor' is provided, we accumulate into it (In-Place).
        # If not, we allocate a fresh one (Standard).
        if out_tensor is None:
            out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
        else:
            out = out_tensor

        # =========================================================
        #  PATH 1: CPU Fallback (Mac / Debugging)
        # =========================================================
        # If Triton is missing (Mac) or the input is on CPU, we use standard PyTorch ops.
        # This ensures the test suite passes on local machines without H100s.
        if not HAS_TRITON or not src.is_cuda:
            # Weighting: [N, D] * [N, 1] -> [N, D]
            weighted = src * weights.unsqueeze(-1)
            # Scatter: Add weighted tokens to their final positions
            # We use index_add_ which is the CPU equivalent of our kernel.
            out.index_add_(0, indices.cpu(), weighted.cpu())
            return out

        # =========================================================
        #  PATH 2: GPU High-Performance Path (Triton)
        # =========================================================
        # Ensure memory layout is contiguous. Triton pointers assume dense packing.
        src = src.contiguous()

        # Safety Check: If the user gave us a buffer, it better be contiguous,
        # otherwise our pointer arithmetic in the kernel will write garbage.
        if not out.is_contiguous():
            raise ValueError("Output tensor must be contiguous")

        N, D = src.shape

        # Heuristic: Block size selection
        # For small hidden dims, use smaller blocks to avoid divergence/masking overhead.
        # For large hidden dims (LLMs usually > 4096), max out the block size.
        BLOCK_SIZE = 1024 if D >= 1024 else 128

        # Grid Size: One program instance per token row.
        grid = (N,)

        # Launch Kernel
        _weighted_scatter_add_kernel[grid](
            src,
            indices,
            weights,
            out,
            src.stride(0),
            out.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            HIDDEN_DIM=D,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward Pass:
        We must compute dL/d(src), dL/d(weights).
        Indices are integers, so they have no gradient.
        """
        indices, weights, src = ctx.saved_tensors

        # Ensure indices are on the correct device for the gather operation.
        # (Handling the edge case where fallback might have moved things).
        indices = indices.to(grad_output.device)

        # 1. Gradient w.r.t Source (Expert Outputs)
        # The output was created by summing sources into specific slots.
        # So, the gradient for a source is simply the gradient of the slot it went to,
        # scaled by the weight it used.
        # Op: Gather gradients from Output -> [N, D]
        grad_output_gathered = grad_output[indices.long()]
        grad_src = grad_output_gathered * weights.unsqueeze(1)

        # 2. Gradient w.r.t Weights (Gating Scores)
        # The weight multiplied the source vector.
        # d(w * src)/dw = src.
        # So grad_w = grad_out * src.
        # Since weight is a scalar per row, we dot-product the vectors and sum.
        grad_weights = (grad_output_gathered * src).sum(dim=1)

        # Return gradients matching the forward signature:
        # (src, indices, weights, out_shape, out_tensor)
        # indices, out_shape, and out_tensor get None.
        return grad_src, None, grad_weights, None, None


def weighted_scatter_add(src, indices, weights, out_shape, out=None):
    """
    Public API for the fused kernel.

    Args:
        src: [N, D] Tensor of expert outputs.
        indices: [N] Tensor of destination indices.
        weights: [N] Tensor of routing weights.
        out_shape: Tuple (Total_Tokens, D).
        out: (Optional) Pre-allocated output tensor for in-place accumulation.
             CRITICAL for the 'Loop over Experts' pattern to avoid allocation overhead.
    """
    return WeightedScatterAdd.apply(src, indices, weights, out_shape, out)
