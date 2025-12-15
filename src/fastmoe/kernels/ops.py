import torch

# =========================================================================
#  Environment Shim
# =========================================================================
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    from unittest.mock import MagicMock

    triton = MagicMock()
    tl = MagicMock()
    triton.jit = lambda f: f
    tl.constexpr = int


@triton.jit
def _grouped_weighted_scatter_add_kernel(
    ptr_table,
    idx_ptr,
    w_ptr,
    out_ptr,
    expert_offsets,
    chunk_sizes,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row_in_expert = tl.program_id(0)
    expert_id = tl.program_id(1)

    n_rows = tl.load(chunk_sizes + expert_id)
    if row_in_expert >= n_rows:
        return

    base_addr_int = tl.load(ptr_table + expert_id)
    src_ptr_base = base_addr_int.to(tl.pointer_type(DTYPE))
    src_row_ptr = src_ptr_base + row_in_expert * HIDDEN_DIM

    global_offset = tl.load(expert_offsets + expert_id)
    global_row_id = global_offset + row_in_expert

    target_row_idx = tl.load(idx_ptr + global_row_id)
    weight = tl.load(w_ptr + global_row_id)

    out_row_ptr = out_ptr + target_row_idx * stride_out_row

    for off in range(0, HIDDEN_DIM, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < HIDDEN_DIM
        val = tl.load(src_row_ptr + cols, mask=mask)
        weighted_val = val * weight
        tl.atomic_add(out_row_ptr + cols, weighted_val, mask=mask)


def prepare_grouped_metadata(expert_tensors, device):
    ptr_list = [t.data_ptr() for t in expert_tensors]
    ptr_table = torch.tensor(ptr_list, dtype=torch.int64, device=device)

    sizes_list = [t.shape[0] for t in expert_tensors]
    chunk_sizes = torch.tensor(sizes_list, dtype=torch.int32, device=device)

    zeros = torch.zeros(1, dtype=torch.int32, device=device)
    cumsum = torch.cumsum(chunk_sizes, dim=0)[:-1]
    expert_offsets = torch.cat((zeros, cumsum))

    max_rows = max(sizes_list) if sizes_list else 0
    num_experts = len(sizes_list)

    return ptr_table, chunk_sizes, expert_offsets, max_rows, num_experts


def _launch_grouped_kernel(expert_tensors, indices, weights, out, metadata=None):
    if metadata is None:
        ptr_table, chunk_sizes, expert_offsets, max_rows, num_experts = prepare_grouped_metadata(
            expert_tensors, indices.device
        )
    else:
        ptr_table, chunk_sizes, expert_offsets, max_rows, num_experts = metadata

    if max_rows > 0:
        N, D = out.shape
        BLOCK_SIZE = 1024 if D >= 1024 else 128
        grid = (max_rows, num_experts)
        triton_dtype = get_triton_dtype(expert_tensors[0].dtype)

        _grouped_weighted_scatter_add_kernel[grid](
            ptr_table,
            indices,
            weights,
            out,
            expert_offsets,
            chunk_sizes,
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            HIDDEN_DIM=D,
            DTYPE=triton_dtype,
        )
    return out


class GroupedWeightedScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weights, out_shape, *expert_tensors):
        # Save everything needed for correct backward (including grad_weights)
        ctx.save_for_backward(indices, weights, *expert_tensors)
        ctx.expert_sizes = [t.shape[0] for t in expert_tensors]

        ref = expert_tensors[0]
        out = torch.zeros(out_shape, device=ref.device, dtype=ref.dtype)
        _launch_grouped_kernel(expert_tensors, indices, weights, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        indices, weights = saved[0], saved[1]
        expert_tensors = saved[2:]
        sizes = ctx.expert_sizes

        # dL/dX = dL/dY[target] * w
        gathered = grad_output.index_select(0, indices)  # [N, D]
        grad_expert_flat = gathered * weights.unsqueeze(-1)  # [N, D]
        grad_experts = torch.split(grad_expert_flat, sizes)

        # dL/dw_i = <dL/dY[target_i], X_i>
        # (use fp32 accumulate for stability; cast back)
        combined = torch.cat(expert_tensors, dim=0)  # [N, D]
        grad_w = (gathered.float() * combined.float()).sum(dim=1).to(weights.dtype)

        # Args: (indices, weights, out_shape, *expert_tensors)
        return None, grad_w, None, *grad_experts


def grouped_weighted_scatter_add(
    expert_tensors, indices, weights, out_shape, out=None, metadata=None
):
    if not indices.is_cuda:
        if out is None:
            out = torch.zeros(out_shape, device=indices.device, dtype=expert_tensors[0].dtype)
        offset = 0
        for t in expert_tensors:
            k = t.shape[0]
            if k > 0:
                idx_chunk = indices[offset : offset + k]
                w_chunk = weights[offset : offset + k]
                out.index_add_(0, idx_chunk, t * w_chunk.unsqueeze(-1))
            offset += k
        return out

    if out is not None:
        return _launch_grouped_kernel(expert_tensors, indices, weights, out, metadata=metadata)

    return GroupedWeightedScatterAdd.apply(indices, weights, out_shape, *expert_tensors)


def weighted_scatter_add(src, indices, weights, out_shape, out=None):
    return grouped_weighted_scatter_add([src], indices, weights, out_shape, out)


def get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    elif torch_dtype == torch.float16:
        return tl.float16
    return tl.float32
