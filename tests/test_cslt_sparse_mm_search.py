import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# cuSPARSELt structured (2:4) sparse matmul search expects CUDA inputs and
# supports the half-precision dtypes that cuSPARSELt is tuned for. float32 is
# excluded because cuSPARSELt's matmul search is not usable there in practice.
SPARSE_MM_DTYPES = [torch.float16, torch.bfloat16]


# Small square/rectangular shapes keep the cuSPARSELt algorithm search fast.
SPARSE_MM_SHAPES = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (64, 128, 64),
    (128, 64, 256),
]


def _make_compressed_and_dense(m, k, n, dtype, device):
    """Build a 2:4 sparse ``compressed_A`` and the matching dense ``B``.

    Returns ``(dense_a_pruned, compressed_a, dense_b)``.
    """
    a = torch.randn(m, k, dtype=dtype, device=device)
    dense_a_pruned = _prune_to_2x4(a)
    compressed_a = torch._cslt_compress(dense_a_pruned)
    b = torch.randn(k, n, dtype=dtype, device=device)
    return dense_a_pruned, compressed_a, b


def _prune_to_2x4(dense):
    """Zero out the 2 smallest-magnitude values in every group of 4 along K.

    Returns a dense tensor that satisfies the 2:4 structured-sparsity pattern so
    it can be compressed with ``torch._cslt_compress``.
    """
    m, k = dense.shape
    # Reshape K into groups of 4 and keep the 2 largest-magnitude entries.
    grouped = dense.reshape(m, k // 4, 4)
    abs_vals = grouped.abs()
    # ranks within each group: the 2 smallest abs values get pruned (masked 0).
    _, top_idx = abs_vals.topk(2, dim=-1)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, top_idx, True)
    pruned = grouped * mask
    return pruned.reshape(m, k)


@pytest.mark.cslt_sparse_mm_search
@pytest.mark.parametrize("shape", SPARSE_MM_SHAPES)
@pytest.mark.parametrize("dtype", SPARSE_MM_DTYPES)
def test_cslt_sparse_mm_search(shape, dtype):
    """The search op must return a valid cuSPARSELt algorithm id.

    Correctness is validated functionally: the id returned by the Gems
    implementation, when fed to ``torch._cslt_sparse_mm`` as ``alg_id``, must
    reproduce the dense reference matmul (computed on the pruned sparse A).
    Multiple distinct ids can be valid, so we do not compare the id itself to
    PyTorch's; we compare the matmul result instead.
    """
    m, k, n = shape
    dense_a_pruned, compressed_a, dense_b = _make_compressed_and_dense(
        m, k, n, dtype, flag_gems.device
    )

    with flag_gems.use_gems():
        alg_id = torch._cslt_sparse_mm_search(compressed_a, dense_b)

    # The result is a Python int (the aten op returns int).
    assert isinstance(alg_id, int), f"expected int, got {type(alg_id)}"

    # Use the searched id to actually run the sparse matmul and verify it
    # matches the dense reference computed from the pruned sparse matrix.
    res_out = torch._cslt_sparse_mm(compressed_a, dense_b, alg_id=alg_id)
    ref_out = torch.mm(dense_a_pruned, dense_b)

    # cuSPARSELt selects a numerically valid but not necessarily bit-identical
    # algorithm; compare in the native half-precision dtype with a permissive
    # absolute tolerance (cf. tests/test_sparse_semi_structured_mm.py).
    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.05)


@pytest.mark.cslt_sparse_mm_search
@pytest.mark.parametrize("dtype", SPARSE_MM_DTYPES)
def test_cslt_sparse_mm_search_with_bias(dtype):
    """The search op honors the ``bias`` argument.

    The searched id (with ``bias``) is fed back to ``torch._cslt_sparse_mm``
    with the same ``bias``. Because cuSPARSELt fuses the bias epilogue with
    half-precision accumulation, the bit pattern is not stable against a
    dense ``torch.mm`` reference (especially for bfloat16). We therefore
    validate consistency against the native cuSPARSELt path: both the Gems and
    the native search select a valid algorithm, and the resulting matmuls must
    agree to the cuSPARSELt precision for this dtype.
    """
    m, k, n = 128, 128, 64
    _, compressed_a, dense_b = _make_compressed_and_dense(
        m, k, n, dtype, flag_gems.device
    )
    bias = torch.randn(n, dtype=dtype, device=flag_gems.device)

    ref_alg_id = torch._cslt_sparse_mm_search(compressed_a, dense_b, bias=bias)
    with flag_gems.use_gems():
        alg_id = torch._cslt_sparse_mm_search(compressed_a, dense_b, bias=bias)

    assert isinstance(alg_id, int)

    ref_out = torch._cslt_sparse_mm(compressed_a, dense_b, bias=bias, alg_id=ref_alg_id)
    res_out = torch._cslt_sparse_mm(compressed_a, dense_b, bias=bias, alg_id=alg_id)
    assert torch.isfinite(
        res_out
    ).all(), "sparse mm with bias produced non-finite values"

    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.05)


@pytest.mark.cslt_sparse_mm_search
@pytest.mark.parametrize("dtype", SPARSE_MM_DTYPES)
def test_cslt_sparse_mm_search_with_alpha(dtype):
    """The search op honors the ``alpha`` scaling argument.

    Validated against the native cuSPARSELt path (see
    ``test__cslt_sparse_mm_search_with_bias`` for rationale): both searches
    return a valid algorithm and the fused matmuls must agree.
    """
    m, k, n = 128, 128, 128
    _, compressed_a, dense_b = _make_compressed_and_dense(
        m, k, n, dtype, flag_gems.device
    )
    alpha = torch.tensor(2.0, dtype=torch.float32, device=flag_gems.device)

    ref_alg_id = torch._cslt_sparse_mm_search(compressed_a, dense_b, alpha=alpha)
    with flag_gems.use_gems():
        alg_id = torch._cslt_sparse_mm_search(compressed_a, dense_b, alpha=alpha)

    assert isinstance(alg_id, int)

    ref_out = torch._cslt_sparse_mm(
        compressed_a, dense_b, alpha=alpha, alg_id=ref_alg_id
    )
    res_out = torch._cslt_sparse_mm(compressed_a, dense_b, alpha=alpha, alg_id=alg_id)
    assert torch.isfinite(
        res_out
    ).all(), "sparse mm with alpha produced non-finite values"

    # alpha scales the output, so widen the absolute tolerance accordingly.
    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.1)
