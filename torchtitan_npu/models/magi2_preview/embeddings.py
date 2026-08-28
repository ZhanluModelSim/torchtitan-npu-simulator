# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview positional embeddings: element-wise Fourier (RoPE) features
and 1-D sinusoidal timestep embeddings.

Fork reason: MAGI-2-preview RoPE features are built from packed (t, h, w,
sizes, refs) coordinates with per-axis rescaling; upstream torchtitan has no
equivalent.
Reference: inference/model/magi2_preview.py::ElementWiseFourierEmbed and
sinusoidal_embedding_1d.
"""

import math

import torch
from torch import nn


class ElementWiseFourierEmbed(nn.Module):
    """Element-wise Fourier embedding for packed 3-D grid coordinates.

    Maps coords ``(T, 9)`` with column order ``(t, h, w, T, H, W, ref_T,
    ref_H, ref_W)`` to ``(T, 6 * num_bands)`` sin/cos features, where
    ``num_bands = dim // 8`` (``dim`` is the per-head rotary dim: 128 for the
    full model gives 16 bands and a 96-dim embedding applied to the first 96
    of 128 head dims).

    Per-axis scale is ``(ref - 1) / (size - 1)``; axes with ``ref == size ==
    1`` are forced to scale 1 to avoid 0/0. Only ``h``/``w`` are centered by
    ``(size - 1) / 2``; ``t`` is not centered.

    The ``bands`` buffer is persistent (checkpoint key ``pre_adapter.rope.bands``)
    and is filled with ``temperature ** (-i / num_bands)`` by
    ``reset_parameters`` / the model-level ``init_weights``.

    Args:
        dim: per-head rotary dim; the band count is ``dim // 8``.
        temperature: inverse-frequency temperature.
    """

    __constants__ = ["dim", "temperature", "num_bands"]

    dim: int
    temperature: float
    num_bands: int

    def __init__(self, dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.num_bands = dim // 8
        self.register_buffer(
            "bands", torch.empty(self.num_bands, dtype=torch.float32)
        )

    def reset_parameters(self) -> None:
        with torch.no_grad():
            exp = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
            self.bands.copy_(torch.exp(-math.log(self.temperature) * exp))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_xyz = coords[:, :3]  # (t, h, w)
        sizes = coords[:, 3:6]  # (T, H, W)
        refs = coords[:, 6:9]  # (ref_T, ref_H, ref_W)

        # per-axis scaling factor; (ref == 1, size == 1) axes keep scale 1.
        # The denominator is guarded before the division so the 0/0 axes
        # produce no NaNs in values or gradients.
        unit = (refs == 1) & (sizes == 1)
        denom = torch.where(unit, torch.ones_like(sizes), sizes - 1)
        scales = torch.where(unit, torch.ones_like(refs), (refs - 1) / denom)

        # center only the h, w dims; leave t un-centered
        coords_xyz = torch.cat(
            [coords_xyz[:, :1], coords_xyz[:, 1:3] - (sizes[:, 1:3] - 1) / 2], dim=1
        )

        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((proj.sin(), proj.cos()), dim=1).flatten(1)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """1-D sinusoidal timestep embedding of shape ``(T, dim)``.

    Positions are scaled by 1000 and paired with inverse-frequency bands;
    output column order is ``[cos, sin]``. For MAGI-2-preview the positions
    are per-token flow-matching sigmas (0 for text tokens) and ``dim=64``.

    Args:
        dim: output dim (rounded up to even internally).
        position: 1-D tensor of per-token positions.
    """
    position = position.to(torch.float32) * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=position.device)
        / half
    )
    args = position[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
