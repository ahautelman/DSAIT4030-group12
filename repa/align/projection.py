from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class TokenProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TokenProjector expects [B, N, C], got {tuple(x.shape)}")
        return self.mlp(x)


class ConvProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ConvProjector expects [B, C, H, W], got {tuple(x.shape)}")
        return self.conv(x)
    
def _extract_teacher_channels(teacher_example: torch.Tensor) -> int:
    if teacher_example.ndim == 3:
        return teacher_example.shape[-1]
    if teacher_example.ndim == 4:
        return teacher_example.shape[1]
    raise ValueError(
        f"teacher_example must have shape [B, N, C] or [B, C, H, W], got {tuple(teacher_example.shape)}"
    )


def build_projection_for_student(
    meta,
    student_feature_sample: torch.Tensor,
    teacher_example: torch.Tensor,
    mode: str = "repa",
) -> nn.Module:
    if student_feature_sample is None:
        raise RuntimeError("student_feature_sample is required to build the lazy projection head.")
    if teacher_example is None:
        raise RuntimeError("teacher_example is required to build the lazy projection head.")

    mode = mode.lower()
    teacher_channels = _extract_teacher_channels(teacher_example)

    is_spatial_mode = mode in ["irepa", "dog"]
    kernel_size = 3 if is_spatial_mode else 1

    if meta.feature_shape_hint == "tokens":
        if student_feature_sample.ndim != 3:
            raise ValueError(
                f"Token-shaped student features must have shape [B, N, C], got {tuple(student_feature_sample.shape)}"
            )
        student_channels = student_feature_sample.shape[-1]
        if is_spatial_mode:
            return ConvProjector(student_channels, teacher_channels, kernel=kernel_size)
        else:
            return TokenProjector(student_channels, teacher_channels)

    if meta.feature_shape_hint == "spatial":
        if student_feature_sample.ndim != 4:
            raise ValueError(
                f"Spatial-shaped student features must have shape [B, C, H, W], got {tuple(student_feature_sample.shape)}"
            )
        student_channels = student_feature_sample.shape[1]
        return ConvProjector(student_channels, teacher_channels, kernel=1)

    raise ValueError(f"Unsupported feature_shape_hint: {meta.feature_shape_hint!r}")
