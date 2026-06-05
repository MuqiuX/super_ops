import torch
import triton
import triton.language as tl
import logging
from collections import namedtuple
import pytest
import triton.experimental.tle.language.gpu as tle_gpu

logger = logging.getLogger(__name__)

def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

@triton.jit
def _uint_to_float(x):
    bits = x.dtype.primitive_bitwidth
    pm: tl.constexpr = 1 << (bits - 1)
    nm: tl.constexpr = (1 << bits) - 1

    mask = tl.where((pm & x) == 0, nm, pm)

    return mask ^ x

@triton.jit
def _float_to_uint(x_bits):
    bits = x_bits.dtype.primitive_bitwidth
    pm: tl.constexpr = 1 << (bits - 1)
    nm: tl.constexpr = (1 << bits) - 1

    udtype = tl.dtype(f"uint{bits}")
    ux = x_bits.to(udtype, bitcast=True)

    mask = tl.where((pm & ux) == 0, pm, nm)

    return mask ^ ux

@triton.jit
def _radix_select_kth_kernel(
    x_ptr, out_val_ptr, out_idx_ptr,
    N,
    N_PASSES: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1

    x_type = x_ptr.dtype.element_ty
    x_nbits: tl.constexpr = x_type.primitive_bitwidth
    x_utype = tl.dtype(f"unit{x_nbits}")

    k_to_find = tl.full((), (N + 1) // 2, tl.int32)
    desired = tl.zeros((), dtype=x_utype)
    desired_mask = tl.zeros((), dtype=x_utype)
    row_ptr = x_ptr + pid_m * N

    hist = tle_gpu.alloc([RADIX_SIZE], dtype=tl.int32, scope=tle_gpu.smem, nv_mma_shared_layout=False)
    hist_offset = tl.arange(0, RADIX_SIZE)
    local_ptrs = tle_gpu.local_ptr(hist, (hist_offset,))

    # radix select and prefix match
    for pass_idx in tl.static_range(N_PASSES):
        bit_offset = (N_PASSES - 1 - pass_idx) * RADIX_BITS

        # 清零hist
        tl.store(local_ptrs, 0, mask=hist_offset < RADIX_SIZE)

        for n_start in range(0, N, BLOCK_N):
            offsets = n_start + tl.arange(0, BLOCK_N)
            mask = offsets < N

            vals = tl.load(row_ptr + offsets, mask=mask, other=0.0)
            vals_bits = vals.to(x_utype, bitcast=True)
            vals_u = _float_to_uint(vals_bits)

            match = mask & ((vals_u & desired_mask) == desired)
            keys = (vals_u >> bit_offset) & RADIX_MASK

            count_ptrs = tle_gpu.local_ptr(hist, (keys,))
            tl.atomic_add(count_ptrs, 1, mask=match, sem="relaxed", scope="cta")

        key = tl.full((), 0, dtype=x_utype)
        found = tl.full((), False, dtype=tl.int1)

        for bin_idx in tl.static_range(RADIX_SIZE):
            count = tl.load(tle_gpu.local_ptr(hist, (bin_idx,)))
            take_this_bin = ~found & (k_to_find <= count)
            key = tl.where(take_this_bin, bin_idx, key)
            k_to_find = tl.where(found | take_this_bin, k_to_find, k_to_find - count)
            found = found | take_this_bin

        desired |= key.to(x_utype) << bit_offset
        desired_mask |= RADIX_MASK << bit_offset

    found_idx = N
    desired_val = _uint_to_float(key).to(x_type)
    for n_start in range(0, N, BLOCK_N):
        offsets = n_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        vals = tl.load(row_ptr + offsets, mask=mask, other=0)
        eq_mask = mask & (vals_u == desired_val)
        masked_idx = tl.where(eq_mask, offsets, N)
        min_idx_tiled = tl.min(masked_idx)
        found_idx = tl.where(min_idx_tiled < found_idx, min_idx_tiled, found_idx)

    safe_idx = tl.minimum(found_idx, N - 1)
    result_val = tl.load(row_ptr + safe_idx)

    tl.store(out_val_ptr + pid_m, result_val)
    tl.store(out_idx_ptr + pid_m, found_idx)

def median_dim(inp, dim=-1, keepdim=False):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    M = inp.numel() // N
    shape[dim] = 1

    if inp.ndim > 2:
        inp = inp.reshape(-1, N)

    values = torch.zeros(shape, dtype=inp.dtype, device=inp.device)
    indices = torch.zeros(shape, dtype=torch.int64, device=inp.device)

    num_bits = inp.itemsize * 8
    radix_bits = 8
    n_passes = triton.cdiv(num_bits, radix_bits)
    block_n = 1024

    _radix_select_kth_kernel[(M,)](
        inp, values, indices,
        N,
        n_passes, radix_bits, block_n
    )

    values = torch.reshape(values, shape)
    indices = torch.reshape(indices, shape)

    if not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)
    
    median_out = namedtuple("median", ["values", "indices"])
    out = median_out(values=values, indices=indices)
    return out

@pytest.mark.median
def test_median_dim():
    x = torch.randn((32, 128), dtype=torch.float32, device='cuda')
    print(x)
    print(median_dim(x))