# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: provide torch-compatible and CANN fused rotary embeddings.

The workaround variant indexes complex caches through real views; the CANN
fused variants call ``torch_npu.npu_rotary_mul``.  Select one per RoPE config.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.rope import (
    ComplexRoPE,
    CosSinRoPE,
    _maybe_wrap_positions,
    _reshape_for_broadcast,
)


class _WorkaroundComplexCacheMixin:
    """Replace complex cache indexing without changing its frequencies."""

    cache: torch.Tensor

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Index the complex cache through its real view for Ascend NPU
        # support.
        positions = _maybe_wrap_positions(positions, query)
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        cache_flat = torch.view_as_real(self.cache).flatten(-2, -1)
        reshaped = _reshape_for_broadcast(cache_flat, complex_query_shape, positions)
        reshaped = reshaped.view(*reshaped.shape[:-1], -1, 2)
        return torch.view_as_complex(reshaped)


class WorkaroundComplexRoPE(_WorkaroundComplexCacheMixin, ComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="Torch-compatible ComplexRoPE indexing via view_as_real (workaround)",
)
def workaround(cfg: ComplexRoPE.Config) -> WorkaroundComplexRoPE.Config:
    return derive(cfg, WorkaroundComplexRoPE.Config)


class _FirstRowPositionsMixin:
    """Use the first position row for the fused batch-wide cosine/sine table.

    All batch rows must have the same position layout.
    """

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is not None and positions.size(0) > 1:
            positions = positions[0].unsqueeze(0)
        return super()._reshape_cache(  # pyrefly: ignore [missing-attribute]
            query, positions
        )


class CANNComplexRoPE(_FirstRowPositionsMixin, WorkaroundComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(WorkaroundComplexRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(  # pyrefly: ignore [bad-override]
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cos = rope_cache.real.repeat_interleave(2, dim=-1).to(query.dtype)
        sin = rope_cache.imag.repeat_interleave(2, dim=-1).to(query.dtype)
        if inverse:
            sin = -sin
        xq_out = torch_npu.npu_rotary_mul(
            query.float(), cos, sin, rotary_mode="interleave"
        ).type_as(query)
        if key is None:
            return xq_out
        xk_out = torch_npu.npu_rotary_mul(
            key.float(), cos.to(key.dtype), sin.to(key.dtype), rotary_mode="interleave"
        ).type_as(key)
        return xq_out, xk_out


class CANNCosSinRoPE(_FirstRowPositionsMixin, CosSinRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            raise NotImplementedError("CosSinRoPE does not support inverse rotation.")
        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim].to(query.dtype)
        sin = rope_cache[..., head_dim:].to(query.dtype)
        xq_out = torch_npu.npu_rotary_mul(
            query.float(), cos, sin, rotary_mode="half"
        ).type_as(query)
        xk_out = torch_npu.npu_rotary_mul(
            key.float(), cos.to(key.dtype), sin.to(key.dtype), rotary_mode="half"
        ).type_as(key)
        return xq_out, xk_out


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="CANN fused ComplexRoPE via torch_npu.npu_rotary_mul (interleave mode)",
)
def cann_complex(cfg: ComplexRoPE.Config) -> CANNComplexRoPE.Config:
    return derive(cfg, CANNComplexRoPE.Config)


@override(
    target=CosSinRoPE.Config,
    description="CANN fused CosSinRoPE via torch_npu.npu_rotary_mul (half mode)",
)
def cann_cossin(cfg: CosSinRoPE.Config) -> CANNCosSinRoPE.Config:
    return derive(cfg, CANNCosSinRoPE.Config)
