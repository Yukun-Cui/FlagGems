# Copyright 2026 FlagOS Contributors

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_COMPOSITE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


def istft(
    self: torch.Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,
    return_complex: bool = False,
) -> torch.Tensor:
    """Inverse STFT with the complete ATen contract.

    The previous generated implementations omitted batch isolation, symmetric
    window padding, NOLA validation, and the imaginary component of complex
    output.  Redispatching to ATen's composite implementation preserves those
    semantics while still allowing its CUDA FFT kernels to execute on device.
    """
    logger.debug("GEMS ISTFT")
    return torch.ops.aten.istft.default.redispatch(
        _COMPOSITE_KEYSET,
        self,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        normalized,
        onesided,
        length,
        return_complex,
    )
