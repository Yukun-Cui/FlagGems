import pytest
import torch

import flag_gems

from . import base

# Shapes for the second-order (double) backward of a 2D convolution.
# Each tuple is
#   (N, Cin, H, W, Cout, kH, kW, stride, padding, dilation, groups, transposed, output_padding)
# and the weight layout follows the chosen ``transposed`` flag.
CONV_DOUBLE_BACKWARD_SHAPES = [
    # non-transposed convolutions
    (16, 32, 64, 64, 64, 3, 3, 1, 1, 1, 1, False, (0, 0)),
    (16, 32, 64, 64, 64, 3, 3, 2, 1, 1, 1, False, (0, 0)),
    (8, 64, 32, 32, 128, 3, 3, 2, 1, 1, 1, False, (0, 0)),
    (8, 32, 32, 32, 32, 3, 3, 1, 1, 1, 2, False, (0, 0)),
    # transposed convolutions (weight is (Cin, Cout//groups, kH, kW))
    (8, 64, 16, 16, 32, 3, 3, 2, 1, 1, 1, True, (0, 0)),
    (8, 64, 16, 16, 32, 3, 3, 2, 1, 1, 1, True, (1, 1)),
    (8, 32, 16, 16, 32, 3, 3, 2, 1, 1, 2, True, (0, 0)),
]


class ConvDoubleBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CONV_DOUBLE_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for (
            n,
            cin,
            h,
            w,
            cout,
            kh,
            kw,
            stride,
            padding,
            dilation,
            groups,
            transposed,
            output_padding,
        ) in self.shapes:
            x = torch.randn((n, cin, h, w), dtype=cur_dtype, device=self.device)
            if not transposed:
                weight_shape = (cout, cin // groups, kh, kw)
            else:
                weight_shape = (cin, cout // groups, kh, kw)
            weight = torch.randn(weight_shape, dtype=cur_dtype, device=self.device)
            bias = torch.randn(cout, dtype=cur_dtype, device=self.device)
            if not transposed:
                y = torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            else:
                y = torch.nn.functional.conv_transpose2d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    output_padding=output_padding,
                    groups=groups,
                )
            gO = torch.randn_like(y)
            ggI = torch.randn_like(x)
            ggW = torch.randn_like(weight)
            ggb = torch.randn_like(bias)
            output_mask = (True, True, True)
            yield (
                ggI,
                ggW,
                ggb,
                gO,
                weight,
                x,
                [stride, stride],
                [padding, padding],
                [dilation, dilation],
                transposed,
                list(output_padding),
                groups,
                output_mask,
            )


@pytest.mark.convolution_double_backward
def test_convolution_double_backward():
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvDoubleBackwardBenchmark(
        op_name="convolution_double_backward",
        torch_op=torch.ops.aten._convolution_double_backward,
        # Restricted dtype list per the worktree generator (numerical-stability / precision scope).
        dtypes=[torch.float32, torch.float16],
    )
    bench.set_gems(flag_gems._convolution_double_backward)
    bench.run()
