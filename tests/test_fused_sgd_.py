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

# The two registered ATen variants are ``aten::_fused_sgd`` (out-of-place /
# functional) and ``aten::_fused_sgd_`` (in-place).  pytest forbids marker
# names that start with ``_``, so the leading-underscore ATen names cannot be
# used as marks; the stripped, pytest-legal marks (``fused_sgd`` /
# ``fused_sgd_``) are applied to the test functions below, matching the
# ``fused_adam`` convention.

# SGD fused optimizer tensors are exercised on the flag_gems device (CUDA).
# Several parameter combinations drive the kernel through every branch:
#   - plain momentum update
#   - dampening (1 - tau) factor
#   - nesterov look-ahead
#   - L2 weight decay
#   - maximize (gradient ascent)
#   - is_first_step (momentum buffer initialisation)
# The native CUDA fused-SGD kernel requires momentum > 0, so every case
# below keeps momentum strictly positive to remain comparable to the
# reference implementation.
SGD_CASES = [
    # (weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step)
    (0.0, 0.9, 0.1, 0.0, False, False, False),  # plain momentum
    (0.0, 0.9, 0.1, 0.0, True, False, False),  # nesterov
    (0.01, 0.9, 0.1, 0.0, False, False, False),  # weight decay
    (0.01, 0.9, 0.1, 0.0, False, True, False),  # maximize + weight decay
    (0.0, 0.9, 0.1, 0.5, False, False, False),  # dampening
    (0.01, 0.9, 0.1, 0.0, False, False, True),  # first step (init momentum buffer)
    (0.01, 0.9, 0.05, 0.2, True, True, False),  # everything together
]

SHAPES = utils.POINTWISE_SHAPES


def _skip_if_cpu_ref():
    """Skip when the CPU reference path is requested.

    ``torch._fused_sgd_`` only has a native CUDA fused-SGD implementation, so
    when CI runs the second pass with ``--ref=cpu --quick`` (``utils.TO_CPU``),
    the reference cannot run on CPU. Skip those cases rather than attempting a
    cross-device comparison that cannot succeed.
    """
    if utils.TO_CPU:
        pytest.skip("fused SGD has no native CPU reference (CUDA-only op)")


def _make_inputs(shape, dtype, device, momentum):
    """Build a single-tensor (param, grad, momentum_buffer) triple."""
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    # momentum_buffer is the running buffer; initialise with non-trivial data
    momentum_buf = torch.randn(shape, dtype=dtype, device=device)
    return param, grad, momentum_buf


def _run_ref(op, params, grads, mbufs, **kwargs):
    """Run the reference (native aten) implementation.

    ``torch._fused_sgd_`` dispatches to the FlagGems registered kernel when
    lists are passed (the FlagGems wrapper accepts list inputs), but the native
    CUDA fused-SGD op expects tuple inputs. Passing tuples therefore reaches
    the native path directly, without an explicit dispatch context. The
    reference inputs are routed through ``utils.to_reference`` so the
    cross-device reference path is consistent with the rest of the suite
    (callers guard ``utils.TO_CPU`` so the native CUDA op never runs on CPU).
    """
    ref = lambda xs: [utils.to_reference(x) for x in xs]
    return op(tuple(ref(params)), tuple(ref(grads)), tuple(ref(mbufs)), **kwargs)


def _run_gems(op, params, grads, mbufs, **kwargs):
    """Run the FlagGems implementation directly.

    ``op`` is the FlagGems wrapper (``flag_gems._fused_sgd_`` or
    ``flag_gems._fused_sgd``); calling it directly invokes the Triton kernel
    without going through the aten dispatch machinery.
    """
    return op(params, grads, mbufs, **kwargs)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize(
    "wd,momentum,lr,dampening,nesterov,maximize,is_first_step", SGD_CASES
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fused_sgd_(
    shape, dtype, wd, momentum, lr, dampening, nesterov, maximize, is_first_step
):
    """In-place fused SGD step: compare gems vs native on params/grads/momentum."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    p, g, mb = _make_inputs(shape, dtype, device, momentum)
    # Reference clones
    ref_p = p.clone()
    ref_g = g.clone()
    ref_mb = mb.clone()
    # Gems clones (mutated in-place by the inplace op)
    gems_p = p.clone()
    gems_g = g.clone()
    gems_mb = mb.clone()

    _run_ref(
        torch._fused_sgd_,
        [ref_p],
        [ref_g],
        [ref_mb],
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )
    _run_gems(
        flag_gems._fused_sgd_,
        [gems_p],
        [gems_g],
        [gems_mb],
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )

    utils.gems_assert_close(gems_p, ref_p, dtype)
    utils.gems_assert_close(gems_mb, ref_mb, dtype)
    utils.gems_assert_close(gems_g, ref_g, dtype)


@pytest.mark.fused_sgd_
@pytest.mark.parametrize(
    "wd,momentum,lr,dampening,nesterov,maximize,is_first_step", SGD_CASES
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fused_sgd__multitensor(
    shape, dtype, wd, momentum, lr, dampening, nesterov, maximize, is_first_step
):
    """In-place fused SGD over a list of tensors of differing shapes."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    shapes = [shape, (64, 64), (128,), (7, 7, 7)]
    tensors = [_make_inputs(s, dtype, device, momentum) for s in shapes]
    ref_p = [t[0].clone() for t in tensors]
    ref_g = [t[1].clone() for t in tensors]
    ref_mb = [t[2].clone() for t in tensors]
    res_p = [t[0].clone() for t in tensors]
    res_g = [t[1].clone() for t in tensors]
    res_mb = [t[2].clone() for t in tensors]

    _run_ref(
        torch._fused_sgd_,
        ref_p,
        ref_g,
        ref_mb,
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )
    _run_gems(
        flag_gems._fused_sgd_,
        res_p,
        res_g,
        res_mb,
        weight_decay=wd,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
    )

    for rp, sp, rg, sg, rmb, smb in zip(ref_p, res_p, ref_g, res_g, ref_mb, res_mb):
        utils.gems_assert_close(sp, rp, dtype)
        utils.gems_assert_close(smb, rmb, dtype)
        utils.gems_assert_close(sg, rg, dtype)


@pytest.mark.fused_sgd_
def test_fused_sgd__grad_scale():
    """AMP gradient scaling: grad is divided by grad_scale in-place."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    dtype = torch.float32
    # A single representative size is enough for the grad_scale / found_inf code
    # paths, which are independent of shape and dtype.
    shape = (256, 256)
    p, g, mb = _make_inputs(shape, dtype, device, momentum=0.9)
    ref_p, ref_g, ref_mb = p.clone(), g.clone(), mb.clone()
    res_p, res_g, res_mb = p.clone(), g.clone(), mb.clone()
    gs = torch.tensor([4.0], device=device)

    _run_ref(
        torch._fused_sgd_,
        [ref_p],
        [ref_g],
        [ref_mb],
        weight_decay=0.0,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        grad_scale=gs,
    )
    _run_gems(
        flag_gems._fused_sgd_,
        [res_p],
        [res_g],
        [res_mb],
        weight_decay=0.0,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        grad_scale=gs,
    )

    utils.gems_assert_close(res_p, ref_p, dtype)
    utils.gems_assert_close(res_mb, ref_mb, dtype)
    utils.gems_assert_close(res_g, ref_g, dtype)


@pytest.mark.fused_sgd_
def test_fused_sgd__found_inf():
    """When found_inf is set, the whole step is skipped (no mutation)."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    dtype = torch.float32
    # A single representative size is enough for the found_inf skip path, which
    # is independent of shape and dtype.
    shape = (128, 128)
    p, g, mb = _make_inputs(shape, dtype, device, momentum=0.9)
    fi = torch.tensor([1.0], device=device)
    res_p, res_g, res_mb = p.clone(), g.clone(), mb.clone()

    _run_gems(
        flag_gems._fused_sgd_,
        [res_p],
        [res_g],
        [res_mb],
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        found_inf=fi,
    )

    # Nothing should have changed.
    utils.gems_assert_close(res_p, p, dtype)
    utils.gems_assert_close(res_g, g, dtype)
    utils.gems_assert_close(res_mb, mb, dtype)


@pytest.mark.fused_sgd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fused_sgd(shape, dtype):
    """Functional aten::_fused_sgd must not mutate inputs and returns new tensors."""
    _skip_if_cpu_ref()
    device = flag_gems.device
    p, g, mb = _make_inputs(shape, dtype, device, momentum=0.9)
    p_before = p.clone()
    g_before = g.clone()
    mb_before = mb.clone()

    # The functional variant has no native aten overload, so the reference is
    # the native in-place ``torch._fused_sgd_`` applied to clones (tuples reach
    # the native CUDA fused-SGD op directly). The functional Gems wrapper must
    # reproduce the same result without mutating its inputs.
    ref_p, ref_g, ref_mb = p.clone(), g.clone(), mb.clone()
    _run_ref(
        torch._fused_sgd_,
        [ref_p],
        [ref_g],
        [ref_mb],
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )
    res_out = _run_gems(
        flag_gems._fused_sgd,
        [p],
        [g],
        [mb],
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )

    # Functional must not mutate the caller's tensors.
    assert torch.equal(p, p_before), "functional mutated param in place"
    assert torch.equal(g, g_before), "functional mutated grad in place"
    assert torch.equal(mb, mb_before), "functional mutated momentum buffer in place"

    # The native in-place reference modified ref_p/ref_mb; the functional Gems
    # result must reproduce those values (param and momentum buffer).
    utils.gems_assert_close(res_out[0][0], ref_p, dtype)
    utils.gems_assert_close(res_out[2][0], ref_mb, dtype)
