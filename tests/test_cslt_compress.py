import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The six valid 2:4 patterns expressed as kept-position pairs.
CSLT_PATTERNS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# 2:4 structured sparsity requires a 2-D tensor whose last dimension is a multiple
# of 4 and exactly two non-zero elements per group of 4. cuSPARSELt-style tiles
# also expect the row count to be reasonably aligned, so we use sizes close to
# real 2:4 workloads.
CSLT_SHAPES = (
    [(8, 32)]
    if utils.QUICK_MODE
    else [(8, 16), (16, 16), (32, 64), (64, 128), (64, 512), (128, 1024), (16, 4096)]
)


def _make_24_sparse(x):
    """Zero out the two smallest-magnitude elements of every group of 4."""
    xr = x.reshape(*x.shape[:-1], -1, 4)
    idx = xr.abs().argsort(dim=-1)
    mask = torch.ones_like(xr)
    mask.scatter_(-1, idx[..., :2], 0)
    return (xr * mask).reshape(x.shape)


@pytest.mark.cslt_compress
@pytest.mark.parametrize("shape", CSLT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cslt_compress(shape, dtype):
    res_inp = _make_24_sparse(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    # `torch._cslt_compress` is CUDA-only (cuSPARSELt backend), so the reference
    # stays on GPU via the NoCPU label and is computed with the native op.
    ref_inp = utils.to_reference(res_inp)
    ref_out = torch._cslt_compress(ref_inp)

    with flag_gems.use_gems():
        res_out = torch._cslt_compress(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.cslt_compress
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cslt_compress_all_patterns(dtype):
    """Exercise every one of the six valid 2:4 sparsity patterns per row."""
    M, N = 8, 24  # 6 groups per row -> one of each pattern
    a = torch.zeros((M, N), dtype=dtype, device=flag_gems.device)
    for r in range(M):
        for g, (p0, p1) in enumerate(CSLT_PATTERNS):
            a[r, g * 4 + p0] = (g + 1) * 10 + 0.5
            a[r, g * 4 + p1] = (g + 1) * 10 + 1.5

    ref_inp = utils.to_reference(a)
    ref_out = torch._cslt_compress(ref_inp)

    with flag_gems.use_gems():
        res_out = torch._cslt_compress(a)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.cslt_compress
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cslt_compress_large_dim_tiling(dtype):
    """Cover the case where a single row spans multiple BLOCK_N tiles."""
    res_inp = _make_24_sparse(
        torch.randn((8, 8192), dtype=dtype, device=flag_gems.device)
    )
    ref_inp = utils.to_reference(res_inp)
    ref_out = torch._cslt_compress(ref_inp)

    with flag_gems.use_gems():
        res_out = torch._cslt_compress(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)
