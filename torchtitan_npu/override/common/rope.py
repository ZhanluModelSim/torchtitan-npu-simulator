# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: provide NPU-compatible and fused rotary embeddings.

Compatibility variants index complex caches through real views. Fused variants
also call ``torch_npu.npu_rotary_mul``; select one variant per RoPE config.
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

from torchtitan_npu.patches.torchtitan.models.common.rope import SingleComplexRoPE


def reshape_complex_cache_via_real(
    cache: torch.Tensor,
    query: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    """Index a complex cache through its real view for Ascend NPU support."""
    positions = _maybe_wrap_positions(positions, query)
    complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
    cache_flat = torch.view_as_real(cache).flatten(-2, -1)
    reshaped = _reshape_for_broadcast(cache_flat, complex_query_shape, positions)
    reshaped = reshaped.view(*reshaped.shape[:-1], -1, 2)
    return torch.view_as_complex(reshaped)


class _NPUComplexCacheMixin:
    """Replace complex cache indexing without changing its frequencies."""

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return reshape_complex_cache_via_real(self.cache, query, positions)


class NPUComplexRoPE(_NPUComplexCacheMixin, ComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass


class NPUSingleComplexRoPE(_NPUComplexCacheMixin, SingleComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(SingleComplexRoPE.Config):
        pass


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="NPU-friendly ComplexRoPE indexing via view_as_real",
)
def npu_rope_override(cfg: ComplexRoPE.Config) -> NPUComplexRoPE.Config:
    return derive(cfg, NPUComplexRoPE.Config)


@override(
    target=SingleComplexRoPE.Config,
    exact=True,
    description="NPU-friendly SingleComplexRoPE indexing via view_as_real",
)
def npu_single_complex_rope_override(
    cfg: SingleComplexRoPE.Config,
) -> NPUSingleComplexRoPE.Config:
    return derive(cfg, NPUSingleComplexRoPE.Config)


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
        return super()._reshape_cache(query, positions)


class NPUFusedRoPE(_FirstRowPositionsMixin, NPUComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUComplexRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = rope_cache.real.repeat_interleave(2, dim=-1).to(query.dtype)
        sin = rope_cache.imag.repeat_interleave(2, dim=-1).to(query.dtype)
        xq_out = torch_npu.npu_rotary_mul(
            query.float(), cos, sin, rotary_mode="interleave"
        ).type_as(query)
        xk_out = torch_npu.npu_rotary_mul(
            key.float(), cos.to(key.dtype), sin.to(key.dtype), rotary_mode="interleave"
        ).type_as(key)
        return xq_out, xk_out


class NPUFusedSingleRoPE(_FirstRowPositionsMixin, NPUSingleComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUSingleComplexRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        cos = rope_cache.real.repeat_interleave(2, dim=-1).to(x.dtype)
        sin = rope_cache.imag.repeat_interleave(2, dim=-1).to(x.dtype)
        if inverse:
            sin = -sin
        return torch_npu.npu_rotary_mul(
            x.float(), cos, sin, rotary_mode="interleave"
        ).type_as(x)


class NPUCosSinRoPE(_FirstRowPositionsMixin, CosSinRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    description="NPU fused RoPE via torch_npu.npu_rotary_mul (interleave mode)",
)
def npu_fused_rope_override(cfg: ComplexRoPE.Config) -> NPUFusedRoPE.Config:
    return derive(cfg, NPUFusedRoPE.Config)


@override(
    target=SingleComplexRoPE.Config,
    exact=True,
    description="NPU fused single-tensor RoPE via npu_rotary_mul (interleave)",
)
def npu_fused_single_rope_override(
    cfg: SingleComplexRoPE.Config,
) -> NPUFusedSingleRoPE.Config:
    return derive(cfg, NPUFusedSingleRoPE.Config)


@override(
    target=CosSinRoPE.Config,
    description="NPU fused CosSinRoPE via torch_npu.npu_rotary_mul (half mode)",
)
def npu_cossin_rope_override(cfg: CosSinRoPE.Config) -> NPUCosSinRoPE.Config:
    return derive(cfg, NPUCosSinRoPE.Config)
