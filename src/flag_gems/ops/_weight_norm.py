# Copyright 2026 FlagOS Contributors

import logging

import torch

logger = logging.getLogger(__name__)


def _weight_norm(v: torch.Tensor, g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """ATen entry point for the existing fused weight-normalization path."""
    logger.debug("GEMS _WEIGHT_NORM")
    if v.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"_weight_norm does not support {v.dtype}")
    from flag_gems.fused.weight_norm import weight_norm

    return weight_norm(v, g, dim)
