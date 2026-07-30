# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize(
    "shape, output_size",
    [((2, 3, 7, 9), (3, 4)), ((1, 2, 8, 8), (1, 1)), ((3, 5, 6), (2, 4))],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_adaptive_max_pool2d_backward(shape, output_size, dtype):
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    output, indices = torch.ops.aten.adaptive_max_pool2d(inp, output_size)
    grad_output = torch.randn_like(output)
    expected = torch.ops.aten.adaptive_max_pool2d_backward(grad_output, inp, indices)

    with flag_gems.use_gems(include=["adaptive_max_pool2d_backward"]):
        actual = torch.ops.aten.adaptive_max_pool2d_backward(grad_output, inp, indices)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_adaptive_max_pool2d_backward_grad_input():
    inp = torch.randn((2, 3, 7, 9), device="cuda")
    output, indices = torch.ops.aten.adaptive_max_pool2d(inp, (3, 4))
    grad_output = torch.randn_like(output)
    expected = torch.ops.aten.adaptive_max_pool2d_backward(grad_output, inp, indices)
    actual = torch.empty(0, device="cuda")

    with flag_gems.use_gems(include=["adaptive_max_pool2d_backward_grad_input"]):
        result = torch.ops.aten.adaptive_max_pool2d_backward.grad_input(
            grad_output, inp, indices, grad_input=actual
        )

    assert result is actual
    torch.testing.assert_close(actual, expected)
