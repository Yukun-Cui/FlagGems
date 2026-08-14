import pytest
import torch

from . import base


def _prune_to_2x4(dense):
    """Zero out the 2 smallest-magnitude values in every group of 4 along K."""
    m, k = dense.shape
    grouped = dense.reshape(m, k // 4, 4)
    abs_vals = grouped.abs()
    _, top_idx = abs_vals.topk(2, dim=-1)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, top_idx, True)
    return (grouped * mask).reshape(m, k)


# cuSPARSELt structured (2:4) sparse matmul search is only usable on the
# half-precision dtypes cuSPARSELt is tuned for; float32 search hangs in this
# cuSPARSELt build, so it is excluded from the benchmark.
SPARSE_MM_DTYPES = [torch.float16, torch.bfloat16]


# Small square/rectangular shapes keep the cuSPARSELt algorithm search fast.
SPARSE_MM_SHAPES = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (64, 128, 64),
    (128, 64, 256),
]


class CsltSparseMmSearchBenchmark(base.Benchmark):
    """Benchmark for ``_cslt_sparse_mm_search``.

    The op returns a cuSPARSELt algorithm id (a Python int), so it has no
    bandwidth/TFLOPS interpretation; only latency and speedup are measured.
    """

    def set_shapes(self, shape_file_path=None):
        # cuSPARSELt search latency grows with shape; keep shapes small and
        # ignore the generic core-shapes file (which has 1D / huge shapes that
        # are invalid for a 2:4 sparse matmul search).
        self.shapes = SPARSE_MM_SHAPES
        self.shape_desc = "M, K, N"

    def get_input_iter(self, cur_dtype):
        for m, k, n in self.shapes:
            a = torch.randn(m, k, dtype=cur_dtype, device=self.device)
            compressed_a = torch._cslt_compress(_prune_to_2x4(a))
            dense_b = torch.randn(k, n, dtype=cur_dtype, device=self.device)
            yield compressed_a, dense_b


@pytest.mark.cslt_sparse_mm_search
def test_cslt_sparse_mm_search():
    bench = CsltSparseMmSearchBenchmark(
        op_name="cslt_sparse_mm_search",
        torch_op=torch._cslt_sparse_mm_search,
        dtypes=SPARSE_MM_DTYPES,
    )
    bench.run()
