# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize(
    "shape, output_size",
    [((2, 3, 7, 9), (3, 4)), ((1, 2, 8, 8), (1, 1)), ((3, 5, 6), (2, 4))],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_adaptive_avg_pool2d_backward(shape, output_size, dtype):
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    grad_output = torch.randn((*shape[:-2], *output_size), device="cuda", dtype=dtype)
    expected = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, inp)

    with flag_gems.use_gems(include=["_adaptive_avg_pool2d_backward"]):
        actual = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, inp)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_adaptive_avg_pool2d_backward_out():
    inp = torch.randn((2, 3, 7, 9), device="cuda")
    grad_output = torch.randn((2, 3, 3, 4), device="cuda")
    expected = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, inp)
    actual = torch.empty(0, device="cuda")

    with flag_gems.use_gems(include=["_adaptive_avg_pool2d_backward"]):
        result = torch.ops.aten._adaptive_avg_pool2d_backward.out(
            grad_output, inp, out=actual
        )

    assert result is actual
    torch.testing.assert_close(actual, expected)
