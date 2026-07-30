# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize(
    "block_size, in_features, out_features", [(64, 64, 8), (32, 128, 4)]
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_dyn_quant_pack_4bit_weight(block_size, in_features, out_features, with_bias):
    weights_cpu = torch.randint(
        0, 256, (out_features, in_features // 2), dtype=torch.uint8
    )
    groups = in_features // block_size
    scales_cpu = torch.randn((out_features, groups), dtype=torch.float32)
    bias_cpu = torch.randn(out_features) if with_bias else None
    expected = torch.ops.aten._dyn_quant_pack_4bit_weight(
        weights_cpu,
        scales_cpu,
        bias_cpu,
        block_size,
        in_features,
        out_features,
    )

    weights = weights_cpu.cuda()
    scales = scales_cpu.cuda()
    bias = None if bias_cpu is None else bias_cpu.cuda()
    with flag_gems.use_gems(include=["_dyn_quant_pack_4bit_weight"]):
        actual = torch.ops.aten._dyn_quant_pack_4bit_weight(
            weights, scales, bias, block_size, in_features, out_features
        )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected)


def test_dyn_quant_pack_4bit_weight_rejects_float_weights():
    weights = torch.randn((4, 32), device="cuda")
    scales = torch.randn((4, 1), device="cuda")
    with (
        flag_gems.use_gems(include=["_dyn_quant_pack_4bit_weight"]),
        pytest.raises(RuntimeError, match="uint8"),
    ):
        torch.ops.aten._dyn_quant_pack_4bit_weight(weights, scales, None, 64, 64, 4)
