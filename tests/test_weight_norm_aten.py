# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("shape, dim", [((5, 7), 0), ((5, 7), 1), ((2, 3, 5), 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_weight_norm_aten(shape, dim, dtype):
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    g_shape = [1] * len(shape)
    g_shape[dim] = shape[dim]
    g = torch.randn(g_shape, device="cuda", dtype=dtype, requires_grad=True)

    expected = torch.ops.aten._weight_norm(v, g, dim)
    with flag_gems.use_gems(include=["_weight_norm"]):
        actual = torch.ops.aten._weight_norm(v, g, dim)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
