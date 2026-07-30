# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("shape", [(3, 2, 4, 5), (2, 3, 4, 5, 6), (1, 2, 1, 1, 1)])
@pytest.mark.parametrize(
    "padding", [(0, 0, 0, 0, 0, 0), (1, 2, 0, 2, 3, 1), (-1, 2, 1, 0, 0, 1)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_replication_pad3d_backward(shape, padding, dtype):
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    pl, pr, pt, pb, pf, pk = padding
    output_shape = (
        *shape[:-3],
        shape[-3] + pf + pk,
        shape[-2] + pt + pb,
        shape[-1] + pl + pr,
    )
    grad_output = torch.randn(output_shape, device="cuda", dtype=dtype)
    expected = torch.ops.aten.replication_pad3d_backward(
        grad_output.float(), inp.float(), padding
    ).to(dtype)

    with flag_gems.use_gems(include=["replication_pad3d_backward"]):
        actual = torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_replication_pad3d_backward_grad_input():
    inp = torch.randn((2, 3, 3, 4, 5), device="cuda")
    padding = (1, 2, 1, 2, 1, 1)
    grad_output = torch.randn((2, 3, 5, 7, 8), device="cuda")
    expected = torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)
    actual = torch.empty(0, device="cuda")

    with flag_gems.use_gems(include=["replication_pad3d_backward_grad_input"]):
        result = torch.ops.aten.replication_pad3d_backward.grad_input(
            grad_output, inp, padding, grad_input=actual
        )

    assert result is actual
    torch.testing.assert_close(actual, expected)
