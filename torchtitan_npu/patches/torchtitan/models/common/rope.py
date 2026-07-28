# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

"""Backport ``SingleComplexRoPE`` from TorchTitan PR #3634.

Remove this module after the TorchTitan dependency includes the PR.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.rope import ComplexRoPE

__all__ = ["SingleComplexRoPE"]


class SingleComplexRoPE(ComplexRoPE):
    """Apply complex RoPE to a single tensor."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        rope_cache = self._reshape_cache(x, positions)
        return self.apply_rotary_emb(x, rope_cache, inverse=inverse)

    @staticmethod
    def apply_rotary_emb(
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        if inverse:
            rope_cache = rope_cache.conj()
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * rope_cache).flatten(-2)
        return x_out.type_as(x)
