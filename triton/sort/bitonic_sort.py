import torch
import triton
import triton.language as tl
import pytest
import flag_gems
from triton.language.standard import _log2

@triton.jit
def compare_and_swap(x, indices, flip, step: tl.constexpr, stages: tl.constexpr):
    shape: tl.constexpr = [2 ** step, 2, 2 ** (stages - step - 1)]

    x_ = tl.reshape(x, shape)
    indices_ = tl.reshape(indices, shape)

    mask = tl.arange(0, 2)[None, :, None]
    left_x = tl.broadcast_to(tl.sum(x_ * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right_x = tl.broadcast_to(tl.sum(x_ * mask, 1)[:, None, :], shape).to(x.dtype)
    left_x = tl.reshape(left_x, x.shape)
    right_x = tl.reshape(right_x, x.shape)

    left_indices = tl.broadcast_to(tl.sum(indices_ * (1 - mask), 1)[:, None, :], shape).to(indices.dtype)
    right_indices = tl.broadcast_to(tl.sum(indices_ * mask, 1)[:, None, :], shape).to(indices.dtype)
    left_indices = tl.reshape(left_indices, indices.shape)
    right_indices = tl.reshape(right_indices, indices.shape)

    utype_x = tl.core.get_int_dtype(x.dtype.primitive_bitwidth, False)
    utype_indices = tl.core.get_int_dtype(indices.dtype.primitive_bitwidth, False)

    ileft = left_x.to(utype_x, bitcast=True)
    iright = right_x.to(utype_x, bitcast=True)
    ix = x.to(utype_x, bitcast=True)

    cond = (left_x > right_x) ^ flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))

    ileft_idx = left_indices.to(utype_indices, bitcast=True)
    iright_idx = right_indices.to(utype_indices, bitcast=True)
    ix_idx = indices.to(utype_indices, bitcast=True)
    ret_idx = ix_idx ^ tl.where(cond, ileft_idx ^ iright_idx, tl.zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(indices.dtype, bitcast=True)


@triton.jit
def bitonic_merge(x, indices, stages: tl.constexpr, stage: tl.constexpr, order: tl.constexpr):
    if stage < stages - 1:
        shape: tl.constexpr = [2 ** (stages - 1 - stage), 2, 2 ** stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order

    for step in tl.static_range(stage):
        x, indices = compare_and_swap(x, indices, flip, step + (stages - stage), stages)

    return x, indices

@triton.jit
def bitonic_sort(x_ptr, indices_ptr, N: tl.constexpr, descending: tl.constexpr):
    # 输入必须是2*n长度
    stages: tl.constexpr = _log2(N)

    cols = tl.arange(0, N)
    x = tl.load(x_ptr + cols, mask=cols < N)
    indices = tl.load(indices_ptr + cols, mask=cols < N)

    for stage in tl.static_range(stages):
        # 前stages-1轮交错排序，最后一轮按照要求排序
        x, indices = bitonic_merge(x, indices, stages, stage, 2 if stage < (stages - 1) else descending)
    return x, indices

@pytest.mark.sort
def test_sort():
    x = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)
    indices = torch.arange(8, device=flag_gems.device)
    print(x)
    x, idx = bitonic_sort[(1,)](x, indices, 8, True)
    print(x)
    print(idx)