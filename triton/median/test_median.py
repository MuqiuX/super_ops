import pytest
import torch
from flag_gems.testing import assert_close, assert_equal
import flag_gems

from median import median_dim

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
REDUCTION_SHAPES = [(1, 2), (4096, 256), (1024, 1024, 1024)]
DIM_LIST = [0, -1]
KEEPDIM = [True, False]


@pytest.mark.median
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_accuracy_median_dim(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = inp.to("cpu")
    ref_values, ref_indices = torch.median(ref_inp, dim=dim, keepdim=keepdim)

    res_values, res_indices = median_dim(inp, dim=dim, keepdim=keepdim)

    res_indices = res_indices.to("cpu")
    res_values = res_values.to("cpu")

    assert_close(res=res_values, ref=ref_values, dtype=dtype)
    mask = res_indices != ref_indices
    if mask.any():
        assert_close(res=res_values[mask], ref=ref_values[mask], dtype=dtype)