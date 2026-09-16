# Copyright 2026 FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Generator

import pytest
import torch

import flag_gems

from . import base

# quantized_max_pool1d pools over the last dimension of a 2D (N, L) or 3D
# (N, C, L) quantized input. These shapes mirror typical 1D-conv / pooling
# workloads.
QUANT_POOL_SHAPES = [
    (4, 1024),
    (16, 4096),
    (32, 8192),
    (64, 16384),
    (8, 3, 1024),
    (16, 8, 2048),
]

POOL_PARAMS = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "dilation": 1,
    "ceil_mode": False,
}


def _make_quantized(shape, device):
    fp_tensor = torch.randn(shape, device="cpu").clamp_(-2, 2)
    return torch.quantize_per_tensor(
        fp_tensor, scale=0.1, zero_point=0, dtype=torch.quint8
    ).to(device)


def _out_length(in_l, params):
    effective = (params["kernel_size"] - 1) * params["dilation"] + 1
    return (in_l + 2 * params["padding"] - effective) // params["stride"] + 1


def torch_quantized_max_pool1d(
    q_tensor, kernel_size, stride, padding, dilation, ceil_mode
):
    """Baseline: the aten quantized op itself, which only has a CPU kernel.

    Dequantize + fp32 max_pool1d + requantize would time a different
    computation (two elementwise passes plus a float pool) instead of the
    integer pooling the operator performs, so the honest reference is the same
    op on CPU even though it makes the comparison cross-device.
    """
    return torch.ops.aten.quantized_max_pool1d.default(
        q_tensor.cpu(),
        [kernel_size],
        [stride],
        [padding],
        [dilation],
        ceil_mode,
    )


def torch_quantized_max_pool1d_out(
    q_tensor, kernel_size, stride, padding, dilation, ceil_mode, *, out
):
    return torch.ops.aten.quantized_max_pool1d.out(
        q_tensor.cpu(),
        [kernel_size],
        [stride],
        [padding],
        [dilation],
        ceil_mode,
        out=out.cpu(),
    )


class QuantizedMaxPool1dBenchmark(base.GenericBenchmark):
    # The aten reference is CPU-only, so keep to the hand-picked shapes above
    # rather than letting core_shapes.yaml inject huge ones.
    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANT_POOL_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield _make_quantized(shape, self.device), dict(POOL_PARAMS)


class QuantizedMaxPool1dOutBenchmark(QuantizedMaxPool1dBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            q_tensor = _make_quantized(shape, self.device)
            out_shape = shape[:-1] + (_out_length(shape[-1], POOL_PARAMS),)
            out = torch.quantize_per_tensor(
                torch.zeros(out_shape),
                scale=float(q_tensor.q_scale()),
                zero_point=int(q_tensor.q_zero_point()),
                dtype=q_tensor.dtype,
            ).to(self.device)
            yield q_tensor, {**POOL_PARAMS, "out": out}


@pytest.mark.quantized_max_pool1d
def test_quantized_max_pool1d():
    bench = QuantizedMaxPool1dBenchmark(
        op_name="quantized_max_pool1d",
        input_fn=None,
        torch_op=torch_quantized_max_pool1d,
        dtypes=[torch.quint8],
    )
    bench.set_gems(flag_gems.quantized_max_pool1d)
    bench.run()


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out():
    bench = QuantizedMaxPool1dOutBenchmark(
        op_name="quantized_max_pool1d_out",
        input_fn=None,
        torch_op=torch_quantized_max_pool1d_out,
        dtypes=[torch.quint8],
    )
    bench.set_gems(flag_gems.quantized_max_pool1d_out)
    bench.run()
