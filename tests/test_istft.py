# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("length", [None, 700])
def test_istft_onesided(batched, normalized, length):
    shape = (2, 129, 12) if batched else (129, 12)
    inp = torch.randn(shape, device="cuda", dtype=torch.complex64)
    window = torch.hann_window(256, device="cuda")
    kwargs = dict(
        n_fft=256,
        hop_length=64,
        window=window,
        normalized=normalized,
        length=length,
    )
    expected = torch.istft(inp, **kwargs)

    with flag_gems.use_gems(include=["istft"]):
        actual = torch.istft(inp, **kwargs)

    torch.testing.assert_close(actual, expected)


def test_istft_full_spectrum_complex_output():
    inp = torch.randn((2, 128, 10), device="cuda", dtype=torch.complex64)
    kwargs = dict(n_fft=128, hop_length=32, onesided=False, return_complex=True)
    expected = torch.istft(inp, **kwargs)
    with flag_gems.use_gems(include=["istft"]):
        actual = torch.istft(inp, **kwargs)
    torch.testing.assert_close(actual, expected)


def test_istft_default_window():
    inp = torch.randn((65, 8), device="cuda", dtype=torch.complex64)
    expected = torch.istft(inp, n_fft=128)
    with flag_gems.use_gems(include=["istft"]):
        actual = torch.istft(inp, n_fft=128)
    torch.testing.assert_close(actual, expected)
