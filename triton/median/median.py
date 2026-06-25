import torch
import triton
import triton.language as tl
import logging
from collections import namedtuple
import triton.experimental.tle.language.gpu as tle_gpu
from flag_gems.utils import dim_compress

logger = logging.getLogger(__name__)

@triton.jit
def _get_topmask_and_fullmask(x, x_nbits: tl.constexpr):
    pm: tl.constexpr = 1 << (x_nbits - 1)
    nm: tl.constexpr = (1 << x_nbits) - 1

    # 移位操作会把type强制转化成int32
    true_pm = tl.full(x.shape, pm, dtype=x.dtype)
    true_nm = tl.full(x.shape, nm, dtype=x.dtype)

    return true_pm, true_nm

@triton.jit
def _uint_to_float(x, x_nbits: tl.constexpr):
    pm, nm = _get_topmask_and_fullmask(x, x_nbits)

    mask = tl.where((pm & x) == 0, nm, pm)
    return mask ^ x

@triton.jit
def _float_to_uint(x, x_nbits: tl.constexpr):
    pm, nm = _get_topmask_and_fullmask(x, x_nbits)

    mask = tl.where((pm & x) == 0, pm, nm)

    return mask ^ x

@triton.jit
def _radix_select_kth_kernel(
    x_ptr, out_val_ptr, out_idx_ptr,
    N,
    N_PASSES: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 每一个program处理一行
    pid_m = tl.program_id(0)
    row_ptr = x_ptr + pid_m * N

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1

    # 获取x的类型，位宽以及utype
    x_type = x_ptr.dtype.element_ty
    x_nbits: tl.constexpr = x_type.primitive_bitwidth
    x_utype = tl.core.get_int_dtype(x_nbits, False)

    # 计算中位数2n取n
    k_to_find = tl.full((), (N + 1) // 2, tl.int32)
    # 目标中位数值utype
    desired = tl.zeros((), dtype=x_utype)
    # 目标中位数值mask
    desired_mask = tl.zeros((), dtype=x_utype)

    # 使用share memory作为直方图
    hist = tle_gpu.alloc([RADIX_SIZE], dtype=tl.int32, scope=tle_gpu.smem, nv_mma_shared_layout=False)
    hist_offset = tl.arange(0, RADIX_SIZE)
    local_ptrs = tle_gpu.local_ptr(hist, (hist_offset,))

    # radix select and prefix match
    for pass_idx in tl.static_range(N_PASSES):
        # 当前pass比较的位置offset
        bit_offset = (N_PASSES - 1 - pass_idx) * RADIX_BITS

        # 每轮清空hist
        tl.store(local_ptrs, 0, mask=hist_offset < RADIX_SIZE)

        # 计算直方图
        for n_start in range(0, N, BLOCK_N): #parallel
            offsets = n_start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            vals = tl.load(row_ptr + offsets, mask=mask, other=0.0)

            # 将数值转为对应的utype
            vals_bits = vals.to(x_utype, bitcast=True)
            vals_u = _float_to_uint(vals_bits, x_nbits)

            match = mask & ((vals_u & desired_mask) == desired)
            keys = (vals_u >> bit_offset) & RADIX_MASK

            count_ptrs = tle_gpu.local_ptr(hist, (keys,))
            # Cooperative Thread Array = thread block
            tl.atomic_add(count_ptrs, 1, mask=match, sem="relaxed", scope="cta")

        # 查找中位数所在bin
        key = tl.full((), 0, dtype=x_utype)
        # 查找停止标志位
        found = tl.full((), False, dtype=tl.int1)

        for bin_idx in tl.static_range(RADIX_SIZE): # static_range -》constexper 编译优化直接展开
            count = tl.load(tle_gpu.local_ptr(hist, (bin_idx,)))
            take_this_bin = ~found & (k_to_find <= count)
            key = tl.where(take_this_bin, bin_idx, key)
            k_to_find = tl.where(found | take_this_bin, k_to_find, k_to_find - count)
            found = found | take_this_bin

        desired |= (key.to(x_utype) << bit_offset).to(x_utype)
        desired_mask |= tl.full((), RADIX_MASK << bit_offset, dtype=x_utype)

    found_idx = N
    desired_val = _uint_to_float(desired, x_nbits).to(x_type, bitcast=True)
    for n_start in range(0, N, BLOCK_N):
        offsets = n_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        vals = tl.load(row_ptr + offsets, mask=mask, other=0)
        eq_mask = mask & (vals == desired_val)
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

if __name__ == '__main__':
    x = torch.randn((32, 128), dtype=torch.float16, device='cuda')
    print(torch.median(x, dim=-1))
    print(median_dim(x))
    