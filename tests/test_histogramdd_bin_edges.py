# Copyright 2026 FlagOS Contributors
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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# ``_histogramdd_bin_edges`` is only implemented for the CPU backend in PyTorch,
# so the reference must always be computed on CPU even when TO_CPU is False.
# ``range`` must be a flat tuple of (left, right) pairs for each dimension.

# Pair each (shape, bins) so the bin count always matches the innermost dim.
if utils.QUICK_MODE:
    HIST_CASES = [((64, 2), (5, 3)), ((1024, 3), (5, 3, 4))]
else:
    HIST_CASES = [
        ((10, 1), (1,)),
        ((100, 2), (5, 3)),
        ((1024, 3), (10, 20, 30)),
        ((4096, 4), (4, 2, 6, 8)),
        ((64, 512, 2), (5, 3)),
        ((16, 32, 64, 3), (4, 2, 6)),
    ]
# PyTorch's _histogramdd_bin_edges is CPU-only and only supports float32 on the
# reference path; restrict the dtype sweep accordingly.
HIST_DTYPES = [torch.float32]


def _to_cpu_ref(inp):
    """Move ``inp`` (and any helper structure) to CPU for the reference call."""
    return (
        utils.to_reference(inp).to("cpu")
        if not utils.TO_CPU
        else utils.to_reference(inp)
    )


def _assert_edges_close(res_edges, ref_edges, dtype):
    """Compare two lists of bin-edge tensors produced by the two paths.

    The result tensors live on ``flag_gems.device`` while the reference lives
    on CPU. ``gems_assert_close`` only moves tensors to CPU when one side is
    already on CPU, so we move the CPU reference onto the result device first
    (mirroring ``tests/test_log_sigmoid_forward.py``).
    """
    assert len(res_edges) == len(
        ref_edges
    ), f"len mismatch: {len(res_edges)} != {len(ref_edges)}"
    for d in range(len(ref_edges)):
        res = res_edges[d]
        ref = ref_edges[d]
        if utils.TO_CPU:
            res = res.to("cpu") if res.device.type != "cpu" else res
        else:
            ref = ref.to(res.device) if ref.device.type != res.device.type else ref
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("shape, bins", HIST_CASES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges(shape, bins, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_bin_edges(ref_inp, list(bins))
    res_out = flag_gems._histogramdd_bin_edges(inp, list(bins))

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("shape, bins", HIST_CASES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_with_range(shape, bins, dtype):
    n_dims = shape[-1]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    # Flat (left, right) pairs for each dimension.
    rng = []
    for _ in range(n_dims):
        lo = float(torch.randint(-3, 0, (1,)).item())
        hi = float(torch.randint(1, 4, (1,)).item())
        rng.extend([lo, hi])
    rng = tuple(rng)

    ref_out = torch._histogramdd_bin_edges(ref_inp, list(bins), range=rng)
    res_out = flag_gems._histogramdd_bin_edges(inp, list(bins), range=rng)

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_constant(dtype):
    # When the data is degenerate (min == max) PyTorch widens to a unit range.
    inp = torch.full((100, 2), 5.0, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_bin_edges(ref_inp, [5, 3])
    res_out = flag_gems._histogramdd_bin_edges(inp, [5, 3])

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_explicit_degenerate_range(dtype):
    # An explicit range with left == right is widened to a unit range too.
    inp = torch.randn(100, 2, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    rng = (7.0, 7.0, 0.0, 2.0)
    ref_out = torch._histogramdd_bin_edges(ref_inp, [4, 3], range=rng)
    res_out = flag_gems._histogramdd_bin_edges(inp, [4, 3], range=rng)

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_empty(dtype):
    # An input with no points falls back to the [0, 1] default range.
    inp = torch.empty((0, 2), dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_bin_edges(ref_inp, [5, 3])
    res_out = flag_gems._histogramdd_bin_edges(inp, [5, 3])

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_small_bins(dtype):
    # Edge cases around the bins count: 0 -> single edge, 1 -> [left, right].
    inp = torch.randn(100, 2, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_bin_edges(ref_inp, [0, 1])
    res_out = flag_gems._histogramdd_bin_edges(inp, [0, 1])

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_weight_density(dtype):
    # weight and density must not influence the computed bin edges.
    inp = torch.randn(100, 2, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(100, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)
    ref_weight = _to_cpu_ref(weight)

    ref_out = torch._histogramdd_bin_edges(
        ref_inp, [5, 3], weight=ref_weight, density=True
    )
    res_out = flag_gems._histogramdd_bin_edges(inp, [5, 3], weight=weight, density=True)

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
def test_histogramdd_bin_edges_higher_dim():
    # Inputs with 3+ dimensions are flattened on all but the last axis.
    inp = torch.randn(4, 5, 3, dtype=torch.float32, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_bin_edges(ref_inp, [4, 2, 6])
    res_out = flag_gems._histogramdd_bin_edges(inp, [4, 2, 6])

    _assert_edges_close(res_out, ref_out, torch.float32)


@pytest.mark.histogramdd_bin_edges
def test_histogramdd_bin_edges_validation_errors():
    # Mirror PyTorch's validation error messages.
    # ndim < 2
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        flag_gems._histogramdd_bin_edges(torch.randn(10, device=flag_gems.device), [5])
    # bins length != ndim
    with pytest.raises(RuntimeError, match="size of bins must be equal"):
        flag_gems._histogramdd_bin_edges(
            torch.randn(10, 2, device=flag_gems.device), [5]
        )
    # range length mismatch
    with pytest.raises(RuntimeError, match="should have 6 elements"):
        flag_gems._histogramdd_bin_edges(
            torch.randn(10, 3, device=flag_gems.device),
            [5, 3, 2],
            range=(0.0, 1.0, 2.0),
        )
    # min > max
    with pytest.raises(RuntimeError, match="min should not exceed max"):
        flag_gems._histogramdd_bin_edges(
            torch.randn(10, 3, device=flag_gems.device),
            [5, 3, 2],
            range=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        )
    # non-finite data range
    with pytest.raises(RuntimeError, match="is not finite"):
        flag_gems._histogramdd_bin_edges(
            torch.tensor(
                [[float("inf"), 1.0], [2.0, 3.0], [float("-inf"), 5.0]],
                device=flag_gems.device,
            ),
            [5, 3],
        )


@pytest.mark.histogramdd_bin_edges_out
@pytest.mark.parametrize("shape, bins", HIST_CASES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_bin_edges_out(shape, bins, dtype):
    n_dims = shape[-1]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    bins_list = list(bins)
    ref_out = [torch.empty(0, dtype=dtype) for _ in range(n_dims)]
    torch.ops.aten._histogramdd_bin_edges.out(ref_inp, bins_list, out=ref_out)

    res_out = [
        torch.empty(0, dtype=dtype, device=flag_gems.device) for _ in range(n_dims)
    ]
    flag_gems._histogramdd_bin_edges_out(inp, bins_list, out=res_out)

    _assert_edges_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("nbins", [-3, -2, -1, 0, 1])
def test_histogramdd_bin_edges_negative_bins(nbins):
    """Parity for bins <= 0.

    ATen builds the edges with ``linspace(steps=bins + 1)``, so bins == -1 (zero
    edges) and bins == 0 (one edge) are legal, and only bins <= -2 raises. A
    kernel that clamps the negative edge count to zero would silently accept -2
    and -3.
    """
    dtype = torch.float32
    inp = torch.randn(16, 1, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_exc = None
    try:
        ref = torch.ops.aten._histogramdd_bin_edges(ref_inp, [nbins])
    except RuntimeError as e:
        ref_exc = e

    if ref_exc is not None:
        with pytest.raises(RuntimeError, match="number of steps must be non-negative"):
            flag_gems._histogramdd_bin_edges(inp, [nbins])
    else:
        res = flag_gems._histogramdd_bin_edges(inp, [nbins])
        assert len(res) == len(ref)
        assert res[0].numel() == ref[0].numel() == nbins + 1
        _assert_edges_close(res, ref, dtype)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
@pytest.mark.parametrize("endpoint", [0, 1], ids=["left", "right"])
def test_histogramdd_bin_edges_nonfinite_explicit_range(bad, endpoint):
    """An explicit non-finite range must raise, not produce non-finite edges."""
    inp = torch.randn(16, 2, dtype=torch.float32, device=flag_gems.device)
    rng = [0.0, 1.0, 0.0, 1.0]
    rng[endpoint] = bad

    with pytest.raises(RuntimeError, match="is not finite"):
        flag_gems._histogramdd_bin_edges(inp, [4, 4], range=rng)

    # Same input is rejected by ATen.
    with pytest.raises(RuntimeError, match="is not finite"):
        torch.ops.aten._histogramdd_bin_edges(_to_cpu_ref(inp), [4, 4], range=rng)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("shape", [(3, 0), (0, 0), (2, 5, 0)])
def test_histogramdd_bin_edges_empty_innermost(shape):
    """An empty innermost dimension yields an empty edge list, as in ATen.

    Also guards the reshape(-1, 0) path, which cannot infer a leading extent.
    """
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref = torch.ops.aten._histogramdd_bin_edges(_to_cpu_ref(inp), [])
    res = flag_gems._histogramdd_bin_edges(inp, [])
    assert len(res) == len(ref) == 0


@pytest.mark.histogramdd_bin_edges_out
def test_histogramdd_bin_edges_out_contract():
    """``out=`` parity: list length, dtype, reuse, resize and non-contiguity."""
    dtype = torch.float32
    dev = flag_gems.device
    inp = torch.randn(16, 2, dtype=dtype, device=dev)
    bins = [3, 3]

    # Wrong list length is rejected (ATen raises for 1 and 3 given 2 dimensions).
    for n in (1, 3):
        outs = [torch.empty(0, dtype=dtype, device=dev) for _ in range(n)]
        with pytest.raises(RuntimeError):
            flag_gems._histogramdd_bin_edges_out(inp, bins, out=outs)

    # A mismatched out dtype must raise rather than being silently cast by copy_.
    bad = [torch.empty(0, dtype=torch.float64, device=dev) for _ in range(2)]
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems._histogramdd_bin_edges_out(inp, bins, out=bad)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.ops.aten._histogramdd_bin_edges.out(
            _to_cpu_ref(inp), bins, out=[t.cpu() for t in bad]
        )

    # Wrongly sized outputs are resized in place, reusing the same objects.
    outs = [
        torch.empty(99, dtype=dtype, device=dev),
        torch.empty(1, dtype=dtype, device=dev),
    ]
    ids = [id(o) for o in outs]
    flag_gems._histogramdd_bin_edges_out(inp, bins, out=outs)
    assert [o.numel() for o in outs] == [4, 4]
    assert [id(o) for o in outs] == ids

    ref_out = [torch.empty(0, dtype=dtype) for _ in range(2)]
    torch.ops.aten._histogramdd_bin_edges.out(_to_cpu_ref(inp), bins, out=ref_out)
    _assert_edges_close(outs, ref_out, dtype)

    # A non-contiguous output of the right size is accepted, as in ATen.
    base = torch.empty(8, dtype=dtype, device=dev)
    nc = base[::2]
    assert not nc.is_contiguous() and nc.numel() == 4
    outs = [nc, torch.empty(0, dtype=dtype, device=dev)]
    flag_gems._histogramdd_bin_edges_out(inp, bins, out=outs)
    _assert_edges_close(outs, ref_out, dtype)
