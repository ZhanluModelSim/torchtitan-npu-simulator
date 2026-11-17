# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""External-table adapter for the shared half-split rotary implementation."""

from dataclasses import dataclass

import torch
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.protocols.module import Module


class HalfRotation(Module):
    """Half-split rotation with externally supplied half-width tables.

    The apply half of a ``CosSinRoPE`` for callers that own their position
    tables (per-image 2D grids) and cannot use a position-indexed cache.
    ``cos``/``sin`` carry one entry per frequency (width ``x.shape[-1] // 2``);
    the input is split into real/imag halves along the last dimension.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        if inverse:
            sin = -sin
        cache = torch.cat((cos, cos, sin, sin), dim=-1)
        return CosSinRoPE.apply_rotary_emb(x, None, cache)  # pyrefly: ignore [bad-argument-type, bad-return]
