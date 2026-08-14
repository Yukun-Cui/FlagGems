import pytest
import torch

from . import base, consts

# 2:4 structured-sparse weights close to real semi-structured workloads: the last
# dimension must be a multiple of 4 and the row count reasonably tile-aligned.
CSLT_COMPRESS_SHAPES = [
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 1024),
]


def _make_24_sparse(x):
    """Zero out the two smallest-magnitude elements of every group of 4."""
    xr = x.reshape(*x.shape[:-1], -1, 4)
    idx = xr.abs().argsort(dim=-1)
    mask = torch.ones_like(xr)
    mask.scatter_(-1, idx[..., :2], 0)
    return (xr * mask).reshape(x.shape)


class CsltCompressBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CSLT_COMPRESS_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            dense = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (_make_24_sparse(dense),)


@pytest.mark.cslt_compress
def test_cslt_compress():
    bench = CsltCompressBenchmark(
        op_name="cslt_compress",
        torch_op=torch._cslt_compress,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
