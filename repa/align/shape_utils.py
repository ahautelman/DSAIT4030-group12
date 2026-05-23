from __future__ import annotations

from math import isqrt

import torch
import torch.nn.functional as F


def tokens_to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected token tensor with shape [B, N, C], got {tuple(tensor.shape)}")

    batch_size, num_tokens, channels = tensor.shape
    height = isqrt(num_tokens)
    if height * height != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a perfect square; cannot reshape to a spatial grid.")

    return tensor.transpose(1, 2).contiguous().view(batch_size, channels, height, height)


def spatial_to_flat(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected spatial tensor with shape [B, C, H, W], got {tuple(tensor.shape)}")

    return tensor.flatten(2).transpose(1, 2).contiguous()


def match_teacher_spatial_grid(student_spatial: torch.Tensor, target_spatial: torch.Tensor) -> torch.Tensor:
    if student_spatial.ndim != 4:
        raise ValueError(
            f"student_spatial must have shape [B, C, H, W], got {tuple(student_spatial.shape)}"
        )
    if target_spatial.ndim != 4:
        raise ValueError(
            f"target_spatial must have shape [B, C, H, W], got {tuple(target_spatial.shape)}"
        )

    return F.interpolate(
        student_spatial,
        size=target_spatial.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

