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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# (batch, input_size, hidden_size) shapes for the quantized GRU cell.
# Deliberately mixes powers of two with sizes that are not a multiple of any
# vector width, and hidden sizes on either side of the kernel's BLOCK_H
# (next_power_of_2, min 16) and CHUNK (64) boundaries.
GRU_SHAPES = [
    (1, 8, 4),
    (2, 16, 8),
    (4, 32, 16),
    (8, 64, 32),
    (16, 10, 20),
    (3, 7, 13),  # non-power-of-two sizes
    (5, 65, 17),  # input_size and hidden_size just past CHUNK / BLOCK_H
    (1, 1, 1),  # minimal
    (7, 129, 65),
]

# aten::quantized_gru_cell dispatches to the FBGEMM packed kernel and is
# CPU-only ("matmul is not supported with quantized cell params" on CUDA),
# and its activation must be fp32.  The oracle therefore always runs on CPU
# in fp32; the FlagGems kernel runs on the GPU.
_ATEN_DTYPE = torch.float32

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="Triton kernel is CUDA-only",
)


# ---------------------------------------------------------------------------
# Why this file does not use ``torch.quantized_gru_cell`` as its main oracle.
#
# ``aten::quantized_gru_cell`` is *not* a host-independent function. It
# decomposes into ``fbgemm_linear_int8_weight_fp32_activation``, and FBGEMM
# picks its u8 x i8 GEMM kernel from the runtime CPU ISA. Without AVX512-VNNI
# the kernel is built on ``vpmaddubsw``, which forms the products of two
# adjacent k-lanes in a **saturating int16** before widening to int32; with VNNI
# (``vpdpbusd``) the same products accumulate straight into int32 and cannot
# saturate. So on a pre-VNNI host every int32 accumulator whose adjacent-lane
# pair product leaves [-32768, 32767] silently saturates, and the op returns a
# different number for the same inputs.
#
# Measured on one H20 host (Xeon 6759P, VNNI + AMX) for shape (1, 8, 4), by
# forcing FBGEMM's dispatch with ``FBGEMM_ENABLE_INSTRUCTIONS``::
#
#   AVX512_VNNI / VNNI_256 / unset : [-0.70214, -0.93644, -0.55252, 0.96268]
#   AVX2 / AVX512 / AVX512_256     : [-0.66276, -0.79706, -0.55003, 0.96268]
#
# i.e. 0.139 apart on one element, ~35x any sane fp16 tolerance. Reproducing
# the GEMM in exact Python integers shows the VNNI answer is the arithmetically
# correct one, and that the AVX2 answer is exactly what a saturating-int16 pair
# product predicts: over the full shape x dtype matrix the exact model matched
# the VNNI path on 3168/3168 output elements and the saturating model matched
# the AVX2 path on 3168/3168.
#
# The Triton kernel accumulates in int32 and never saturates, so it agrees with
# the VNNI path and disagrees with the AVX2 path. That is the kernel being
# right, not wrong -- but it means an unconditional comparison against
# ``torch.quantized_gru_cell`` asserts a property of the *test runner's CPU*.
#
# Hence the split below:
#   * ``_reference_gru_cell`` -- the same algorithm with an exact integer
#     accumulator, evaluated in torch. Host-independent, so it carries the
#     shape/dtype/stride coverage.
#   * ``test_quantized_gru_cell_matches_aten`` -- a real cross-check against
#     the aten op, restricted to int8 weights small enough that ``vpmaddubsw``
#     provably cannot saturate, which is exactly when aten is well defined.
# ---------------------------------------------------------------------------

# Largest |int8 weight| for which FBGEMM's pre-VNNI int16 pair product cannot
# saturate: a uint8 activation code is at most 255, and a vpmaddubsw lane holds
# the sum of two products, so the bound is 2 * 255 * w <= 32767, i.e. w <= 64.
# 63 is used to stay clear of the boundary.
_SATURATION_FREE_W = 63


def _make_quantized_weight(weight_float, clamp_to=None):
    """Quantize a float weight the way PyTorch's fused quantized RNN cells do.

    Returns the ``(int8 weight, col_offsets, scale, zero_point)`` tuple from
    ``torch.fbgemm_linear_quantize_weight`` plus the FBGEMM-packed weight the
    aten reference op requires.

    ``clamp_to`` narrows the int8 weights to ``[-clamp_to, clamp_to]`` and
    recomputes ``col_offsets``/``packed`` accordingly, which is how
    :func:`test_quantized_gru_cell_matches_aten` keeps the aten op out of
    FBGEMM's ISA-dependent saturation regime.
    """
    w_int8, col_offsets, scale, zero_point = torch.fbgemm_linear_quantize_weight(
        weight_float
    )
    if clamp_to is not None:
        w_int8 = w_int8.clamp(-clamp_to, clamp_to).contiguous()
        # col_offsets is rowsum(w_q) - zero_point * K, so it has to follow the
        # clamp; feeding aten the stale offsets would bias every output row.
        col_offsets = (
            w_int8.to(torch.int32).sum(dim=1) - zero_point * w_int8.shape[1]
        ).to(torch.int32)
    packed = torch.fbgemm_pack_quantized_matrix(w_int8)
    return w_int8, col_offsets, scale, zero_point, packed


def _f32(value):
    """The fp32-narrowed value of a Python/double scalar, as a 0-dim tensor."""
    return torch.tensor(float(value), dtype=torch.float32)


def _reference_linear(activation, w_q, w_scale, w_zero_point, bias):
    """``fbgemm_linear_int8_weight_fp32_activation`` with an exact accumulator.

    Mirrors FBGEMM's dynamic-quantization path term for term, but accumulates
    in int64 instead of the ISA-dependent int16/int32 mix, so the result is the
    same on any host:

        act_scale, act_zp = ChooseQuantizationParams(activation)
        qa   = clamp(nearbyint(fmaf(a, 1f/act_scale, act_zp)), 0, 255)
        acc  = (qa - act_zp) @ (w_q - w_zp).T          # exact
        out  = acc * (act_scale * w_scale) + bias      # fp32
    """
    scale, zero_point = torch._choose_qparams_per_tensor(activation.float(), False)
    scale = _f32(scale)
    inv_scale = _f32(1.0) / scale
    # FBGEMM quantizes with a single-rounding ``fmaf``. An fp32 x fp32 product
    # needs at most 48 mantissa bits and the sum at most 53, so evaluating in
    # fp64 and narrowing once reproduces the fma exactly -- unlike the two-step
    # fp32 ``a * inv + zp``, which rounds twice and lands on the wrong side of
    # a half-way tie. Measured against codes read back out of FBGEMM itself
    # (identity int8 weight, so no saturation can interfere): 435560/435600
    # for this form against 435488/435600 for the two-step, and all 40 residual
    # disagreements are exact ``k + 0.5`` ties.
    scaled = (
        activation.float().double() * inv_scale.double() + float(zero_point)
    ).float()
    codes = torch.round(scaled).clamp(0.0, 255.0).to(torch.int64)
    centred = codes - int(zero_point)
    weights = w_q.to(torch.int64) - int(w_zero_point)
    acc = centred @ weights.t()
    return acc.to(torch.float32) * (scale * _f32(w_scale)) + bias.float()


def _reference_gru_cell(
    inp, hx, w_ih_q, w_hh_q, b_ih, b_hh, scale_ih, scale_hh, zp_ih, zp_hh
):
    """Host-independent oracle for ``aten::quantized_gru_cell``.

    Two quantized linears (see :func:`_reference_linear`) followed by aten's
    ``GRUCell`` expression order from ``RNN.cpp``.
    """
    gi = _reference_linear(inp, w_ih_q, scale_ih, zp_ih, b_ih)
    gh = _reference_linear(hx, w_hh_q, scale_hh, zp_hh, b_hh)
    hidden_size = hx.shape[1]
    lo, mid = hidden_size, 2 * hidden_size
    r_gate = torch.sigmoid(gi[:, :lo] + gh[:, :lo])
    z_gate = torch.sigmoid(gi[:, lo:mid] + gh[:, lo:mid])
    n_gate = torch.tanh(gi[:, mid:] + gh[:, mid:] * r_gate)
    return (hx.float() - n_gate) * z_gate + n_gate


def _oracle_args(ref_args):
    """Select the subset of the aten argument tuple the oracle needs.

    ``packed_*``/``col_offsets_*`` describe FBGEMM's packed layout and carry no
    information the centred integer form does not already have.
    """
    return (
        ref_args[0],  # input
        ref_args[1],  # hx
        ref_args[2],  # w_ih_q
        ref_args[3],  # w_hh_q
        ref_args[4],  # b_ih
        ref_args[5],  # b_hh
        ref_args[10],  # scale_ih
        ref_args[11],  # scale_hh
        ref_args[12],  # zero_point_ih
        ref_args[13],  # zero_point_hh
    )


def _build_case(
    shape, dtype, seed=42, activation="randn", clamp_w=None, positive_weights=False
):
    """Build one test case. Weights/qparams are always fp32-derived; only the
    activations and biases take ``dtype``.

    ``positive_weights`` draws the float weights from a positive range so their
    int8 codes are all positive too. Combined with ``activation="positive"``
    (whose uint8 zero point is pinned at 0) every product in the reduction has
    the same sign, so the accumulator grows with ``input_size`` instead of
    cancelling -- the only way to push it past fp32's exact-integer range.
    """
    batch_size, input_size, hidden_size = shape
    torch.manual_seed(seed)

    if positive_weights:
        w_ih_float = (
            torch.rand(3 * hidden_size, input_size, dtype=torch.float32) * 4 + 1
        )
        w_hh_float = (
            torch.rand(3 * hidden_size, hidden_size, dtype=torch.float32) * 4 + 1
        )
    else:
        w_ih_float = torch.randn(3 * hidden_size, input_size, dtype=torch.float32)
        w_hh_float = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32)
    w_ih_q, col_ih, scale_ih, zp_ih, packed_ih = _make_quantized_weight(
        w_ih_float, clamp_to=clamp_w
    )
    w_hh_q, col_hh, scale_hh, zp_hh, packed_hh = _make_quantized_weight(
        w_hh_float, clamp_to=clamp_w
    )

    if activation == "randn":
        inp = torch.randn(batch_size, input_size, dtype=torch.float32)
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    elif activation == "constant":
        inp = torch.full((batch_size, input_size), 2.5, dtype=torch.float32)
        hx = torch.full((batch_size, hidden_size), -1.25, dtype=torch.float32)
    elif activation == "zeros":
        inp = torch.zeros(batch_size, input_size, dtype=torch.float32)
        hx = torch.zeros(batch_size, hidden_size, dtype=torch.float32)
    elif activation == "positive":
        inp = torch.rand(batch_size, input_size, dtype=torch.float32) * 4 + 1
        hx = torch.rand(batch_size, hidden_size, dtype=torch.float32) * 4 + 1
    elif activation == "negative":
        inp = -torch.rand(batch_size, input_size, dtype=torch.float32) * 4 - 1
        hx = -torch.rand(batch_size, hidden_size, dtype=torch.float32) * 4 - 1
    elif activation == "tiny":
        # exercises FBGEMM's SMALL_SCALE_THRESHOLD branch in ChooseQuantizationParams
        inp = torch.randn(batch_size, input_size, dtype=torch.float32) * 1e-8
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32) * 1e-8
    elif activation == "outlier":
        inp = torch.randn(batch_size, input_size, dtype=torch.float32)
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        inp.view(-1)[0] = 1e4
        hx.view(-1)[0] = -1e4
    else:  # pragma: no cover - guard against typos in parametrization
        raise ValueError(activation)

    b_ih = torch.randn(3 * hidden_size, dtype=torch.float32)
    b_hh = torch.randn(3 * hidden_size, dtype=torch.float32)

    # The oracle must see the *same values the kernel sees*: round to `dtype`
    # first, then widen back to fp32 for the CPU op. Feeding the oracle the
    # un-rounded fp32 activation instead is not an apples-to-apples test --
    # rounding changes the tensor's min/max, hence the dynamic qparams, hence
    # the whole uint8 code assignment, which inflates the fp16 gap from
    # ~1e-3 to ~7e-2 and the bf16 gap from ~8e-3 to ~1.7e-1 without any
    # kernel error being involved.
    ref_args = (
        inp.to(dtype).float(),
        hx.to(dtype).float(),
        w_ih_q,
        w_hh_q,
        b_ih.to(dtype).float(),
        b_hh.to(dtype).float(),
        packed_ih,
        packed_hh,
        col_ih,
        col_hh,
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )
    dev = flag_gems.device
    res_args = (
        inp.to(dtype).to(dev),
        hx.to(dtype).to(dev),
        w_ih_q.to(dev),
        w_hh_q.to(dev),
        b_ih.to(dtype).to(dev),
        b_hh.to(dtype).to(dev),
        packed_ih.to(dev),
        packed_hh.to(dev),
        col_ih.to(dev),
        col_hh.to(dev),
        scale_ih,
        scale_hh,
        zp_ih,
        zp_hh,
    )
    return ref_args, res_args


# Tolerances measured against ``_reference_gru_cell``, oracle fed the same
# rounded values the kernel sees, over GRU_SHAPES + the large/strided shapes x
# 40 seeds x 3 dtypes (worst observed max-abs-err), and verified identical under
# every FBGEMM_ENABLE_INSTRUCTIONS setting:
#   fp32     1.6e-06  -> 1e-4
#   fp16     9.8e-04  -> 4e-3
#   bfloat16 7.8e-03  -> 2e-2
# fp32 sits ~60x under its budget: the kernel reproduces FBGEMM's uint8
# activation quantization and exact accumulation, so the only residual is the
# sigmoid/tanh implementation difference. The fp16/bf16 budgets are set by the
# activation's own rounding: an fp16 activation carries ~11 mantissa bits, so
# its uint8 code can land one step away from the fp32 value's code.
_ATOL = {torch.float32: 1e-4, torch.float16: 4e-3, torch.bfloat16: 2e-2}


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", GRU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_gru_cell(shape, dtype):
    """Accuracy against the host-independent quantized reference.

    ``aten::quantized_gru_cell`` decomposes into two
    ``fbgemm_linear_int8_weight_fp32_activation`` calls plus the GRU gate
    combination.  The FBGEMM linear *dynamically quantizes its fp32
    activation to uint8* before the int8 GEMM, so the op is not equivalent to
    a float matmul against dequantized weights: modelling it that way drifts
    by up to ~0.9 absolute at (16, 2048, 256).  The kernel reproduces the
    quantization, so a tight tolerance holds -- see ``_ATOL``.

    The oracle is ``_reference_gru_cell`` rather than the aten op itself
    because the aten op is ISA-dependent; see the note at the top of this
    file, and ``test_quantized_gru_cell_matches_aten`` for the cross-check
    against aten in the regime where aten is well defined.
    """
    ref_args, res_args = _build_case(shape, dtype)

    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(_reference_gru_cell(*_oracle_args(ref_args)))

    assert res_out.dtype == dtype
    assert res_out.shape == (shape[0], shape[2])
    # Pass ``dtype`` rather than ``torch.float32``: ``gems_assert_close`` derives
    # ``rtol`` from it via ``RESOLUTION``, and fp32's 1.3e-6 is far too tight for
    # an fp16/bf16 result upcast for comparison.
    utils.gems_assert_close(res_out.cpu(), ref_out, dtype, atol=_ATOL[dtype])


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", GRU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quantized_gru_cell_matches_aten(shape, dtype):
    """Cross-check against ``torch.quantized_gru_cell`` itself.

    Restricted to int8 weights with ``|w| <= 63``. FBGEMM's pre-VNNI u8 x i8
    kernel sums two adjacent-k products into a *saturating* int16, so with
    ``2 * 255 * 63 = 32130 < 32767`` no lane can saturate and the aten op
    becomes a well-defined, host-independent function -- which is the only
    regime in which asserting agreement with it is meaningful.

    Verified by forcing FBGEMM's dispatch: under this clamp the aten output is
    byte-identical across ``FBGEMM_ENABLE_INSTRUCTIONS`` in {AVX2, AVX512,
    AVX512_VNNI, unset} over 40 seeds x 12 shapes x 3 dtypes, whereas without
    it the AVX2 and VNNI paths differ by up to 2.1 absolute.

    This also pins the ``col_offsets`` contract: the clamp forces them to be
    recomputed, and stale offsets bias every output row.
    """
    ref_args, res_args = _build_case(shape, dtype, clamp_w=_SATURATION_FREE_W)
    assert ref_args[2].abs().max() <= _SATURATION_FREE_W

    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(torch.quantized_gru_cell(*ref_args))

    utils.gems_assert_close(res_out.cpu(), ref_out, dtype, atol=_ATOL[dtype])


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize(
    "activation",
    ["constant", "zeros", "positive", "negative", "tiny", "outlier"],
)
@pytest.mark.parametrize("shape", [(4, 32, 16), (3, 7, 13), (1, 1, 1)])
def test_quantized_gru_cell_adversarial_activations(shape, activation):
    """Activation distributions that stress the dynamic quantization.

    ``zeros``/``constant`` give a degenerate min==max range (FBGEMM falls back
    to scale 0.1 / a saturated zero point), ``positive``/``negative`` pin the
    zero point at 0 or 255, and ``tiny`` drives ChooseQuantizationParams into
    its SMALL_SCALE_THRESHOLD rescaling branch.
    """
    ref_args, res_args = _build_case(shape, _ATEN_DTYPE, activation=activation)
    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(_reference_gru_cell(*_oracle_args(ref_args)))
    utils.gems_assert_close(
        res_out.cpu(), ref_out, _ATEN_DTYPE, atol=_ATOL[_ATEN_DTYPE]
    )


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", [(1, 1024, 16), (1, 4096, 16), (1, 8192, 16)])
def test_quantized_gru_cell_accumulator_exceeds_fp32(shape):
    """Exercise the reduction with an accumulator past fp32's exact range.

    ``test_quantized_gru_cell_large_reduction`` does *not* reach that range:
    with ``randn`` activations the signed products cancel, so at input_size
    8192 the accumulator peaks near 2.4e5 -- 70x below 2**24. Correlating the
    signs (all-positive activations, whose uint8 zero point is pinned at 0,
    against all-positive int8 weights) reaches 2.5e7 / 9.9e7 / 1.9e8 for
    K = 1024 / 4096 / 8192, which the assertion below pins.

    Scope, stated honestly: this covers the reduction and the int32 range
    (1.9e8 is well inside 2**31), but it does **not** discriminate an fp32
    accumulator from an exact one. Swapping the kernel's int32 accumulator for
    fp32 was measured to leave every output here bit-identical, because the
    same correlation that inflates the accumulator also drives the
    requantized gate pre-activation to |gi| ~ 1e4, where sigmoid/tanh
    saturate and absorb the dropped low bits. The two requirements are coupled
    through the requantization multiplier (``gi = acc * act_scale * w_scale``),
    so no choice of inputs makes the accumulator large while keeping the gates
    in their sensitive region. The int32 accumulator is therefore asserted by
    construction and by the exact-integer oracle, not by this test alone.
    """
    ref_args, res_args = _build_case(
        shape, _ATEN_DTYPE, activation="positive", positive_weights=True
    )
    # Guard the premise: if the inputs ever stop producing a large accumulator
    # this test silently stops testing anything.
    scale, zero_point = torch._choose_qparams_per_tensor(ref_args[0], False)
    codes = torch.round(ref_args[0].double() / float(scale) + float(zero_point)).clamp(
        0, 255
    ).to(torch.int64) - int(zero_point)
    acc = codes @ (ref_args[2].to(torch.int64) - int(ref_args[12])).t()
    assert acc.abs().max() > 2**24, f"accumulator only reached {acc.abs().max()}"

    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(_reference_gru_cell(*_oracle_args(ref_args)))
    utils.gems_assert_close(
        res_out.cpu(), ref_out, _ATEN_DTYPE, atol=_ATOL[_ATEN_DTYPE]
    )


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", [(4, 1024, 64), (2, 4096, 32), (2, 8192, 16)])
def test_quantized_gru_cell_large_reduction(shape):
    """Large ``input_size``: many reduction chunks and CHUNK-boundary tails.

    This covers the reduction loop at scale (16 to 128 iterations of the
    CHUNK=64 loop). It does *not* stress the accumulator width -- with signed
    randn activations the products cancel and |acc| peaks near 2.4e5, well
    inside fp32's exact range; see
    ``test_quantized_gru_cell_accumulator_exceeds_fp32`` for that.
    """
    ref_args, res_args = _build_case(shape, _ATEN_DTYPE)
    res_out = flag_gems.quantized_gru_cell(*res_args)
    ref_out = utils.to_reference(_reference_gru_cell(*_oracle_args(ref_args)))
    utils.gems_assert_close(
        res_out.cpu(), ref_out, _ATEN_DTYPE, atol=_ATOL[_ATEN_DTYPE]
    )


@pytest.mark.quantized_gru_cell
def test_quantized_gru_cell_non_contiguous():
    """ATen accepts strided operands; so must the kernel.

    The kernel indexes every operand linearly. Before the fix, a stride-2
    ``b_ih``/``b_hh`` was read as contiguous storage -- silently the wrong
    elements, worth ~1.5 absolute error.
    """
    shape = (6, 33, 17)
    ref_args, res_args = _build_case(shape, _ATEN_DTYPE)
    hidden_size = shape[2]
    gates = 3 * hidden_size
    dev = flag_gems.device

    # Strided biases: a stride-2 view of a 2*gates buffer, created on device so
    # that `.to(device)` cannot silently compact it.
    big_ih = torch.randn(2 * gates, device=dev)
    big_hh = torch.randn(2 * gates, device=dev)
    b_ih_nc = big_ih[::2]
    b_hh_nc = big_hh[::2]
    assert not b_ih_nc.is_contiguous()

    # Strided activations and weights.
    inp_nc = torch.randn(shape[1], shape[0], device=dev).T
    hx_nc = torch.randn(hidden_size, shape[0], device=dev).T
    w_ih_nc = res_args[2].T.contiguous().T
    w_hh_nc = res_args[3].T.contiguous().T
    assert not inp_nc.is_contiguous() and not w_ih_nc.is_contiguous()

    res_out = flag_gems.quantized_gru_cell(
        inp_nc,
        hx_nc,
        w_ih_nc,
        w_hh_nc,
        b_ih_nc,
        b_hh_nc,
        *res_args[6:],
    )
    ref_out = utils.to_reference(
        _reference_gru_cell(
            inp_nc.cpu(),
            hx_nc.cpu(),
            w_ih_nc.cpu(),
            w_hh_nc.cpu(),
            b_ih_nc.cpu(),
            b_hh_nc.cpu(),
            ref_args[10],
            ref_args[11],
            ref_args[12],
            ref_args[13],
        )
    )
    utils.gems_assert_close(
        res_out.cpu(), ref_out, _ATEN_DTYPE, atol=_ATOL[_ATEN_DTYPE]
    )


@pytest.mark.quantized_gru_cell
@pytest.mark.parametrize("shape", [(0, 32, 16), (4, 0, 16)])
def test_quantized_gru_cell_zero_sized(shape):
    """Zero batch and zero ``input_size``. ATen accepts both."""
    batch_size, input_size, hidden_size = shape
    dev = flag_gems.device
    torch.manual_seed(0)
    w_hh_float = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32)
    w_hh_q, col_hh, scale_hh, zp_hh, _ = _make_quantized_weight(w_hh_float)

    w_ih_q = torch.zeros(3 * hidden_size, input_size, dtype=torch.int8, device=dev)
    col_ih = torch.zeros(3 * hidden_size, dtype=torch.int32, device=dev)
    empty = torch.empty(0, device=dev)

    out = flag_gems.quantized_gru_cell(
        torch.randn(batch_size, input_size, device=dev),
        torch.randn(batch_size, hidden_size, device=dev),
        w_ih_q,
        w_hh_q.to(dev),
        torch.randn(3 * hidden_size, device=dev),
        torch.randn(3 * hidden_size, device=dev),
        empty,
        empty,
        col_ih,
        col_hh.to(dev),
        0.1,
        scale_hh,
        0,
        zp_hh,
    )
    assert out.shape == (batch_size, hidden_size)
    assert torch.isfinite(out).all()


@pytest.mark.quantized_gru_cell
def test_quantized_gru_cell_validates_shapes():
    """Malformed shapes must raise, not read out of bounds.

    Every case below raises in ATen. Before the fix the kernel silently
    computed from mismatched extents via raw pointer arithmetic.
    """
    shape = (4, 32, 16)
    batch_size, input_size, hidden_size = shape
    _, args = _build_case(shape, _ATEN_DTYPE)
    args = list(args)
    dev = flag_gems.device

    def expect_raises(match, idx, value):
        mutated = list(args)
        mutated[idx] = value
        with pytest.raises(RuntimeError, match=match):
            flag_gems.quantized_gru_cell(*mutated)

    # input / hx rank
    expect_raises("expected 2-D input", 0, torch.randn(input_size, device=dev))
    expect_raises(
        "expected 2-D hx", 1, torch.randn(1, batch_size, hidden_size, device=dev)
    )
    # batch mismatch
    expect_raises("batch size", 1, torch.randn(batch_size + 3, hidden_size, device=dev))
    # weight shapes
    expect_raises("expected w_ih of shape", 2, args[2][: 2 * hidden_size])
    expect_raises("expected w_ih of shape", 2, args[2][:, :16].contiguous())
    expect_raises(
        "expected w_hh of shape", 3, args[3][:, : hidden_size // 2].contiguous()
    )
    expect_raises("expected 2-D w_ih", 2, args[2].flatten())
    # bias shapes
    expect_raises("expected b_ih of size", 4, args[4][: 2 * hidden_size])
    expect_raises("expected b_hh of size", 5, args[5][:hidden_size])
    expect_raises("expected 1-D b_ih", 4, args[4].view(3, hidden_size))
    # device mismatch
    expect_raises("same device", 1, args[1].cpu())
