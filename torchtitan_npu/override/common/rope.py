# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: provide torch-compatible and AscendC fused rotary embeddings.

The workaround variant uses pre-expanded cosine/sine caches; the AscendC fused
variants call ``torch_npu.npu_rotary_mul``. Select one per RoPE config.
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


class _InterleavedCacheMixin:
    def __init__(self, config: ComplexRoPE.Config):
        super().__init__(config)  # pyrefly: ignore [bad-argument-count]
        self._expand_interleaved_cache()

    def _init_self_buffers(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        super()._init_self_buffers(  # pyrefly: ignore [missing-attribute]
            buffer_device=buffer_device
        )
        self._expand_interleaved_cache()

    def _expand_interleaved_cache(self) -> None:
        complex_cache = self.cache
        self.cache = torch.stack(
            (
                complex_cache.real.repeat_interleave(2, dim=-1),
                complex_cache.imag.repeat_interleave(2, dim=-1),
            )
        )

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = _maybe_wrap_positions(positions, query)
        cos_cache, sin_cache = self.cache.unbind(0)
        return (
            _reshape_for_broadcast(cos_cache, query.shape, positions),
            _reshape_for_broadcast(sin_cache, query.shape, positions),
        )


def _apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x_float = x.float()
    rotated = torch.stack(
        (-x_float[..., 1::2], x_float[..., ::2]),
        dim=-1,
    ).flatten(-2)
    return (x_float * cos + rotated * sin).type_as(x)


class WorkaroundComplexRoPE(  # pyrefly: ignore [inconsistent-inheritance]
    _InterleavedCacheMixin, ComplexRoPE
):
    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(  # pyrefly: ignore [bad-override]
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rope_cache
        if inverse:
            sin = -sin
        query_out = _apply_interleaved_rope(query, cos, sin)
        if key is None:
            return query_out
        return query_out, _apply_interleaved_rope(key, cos, sin)


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="Torch-compatible interleaved ComplexRoPE (workaround)",
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if positions is not None and positions.size(0) > 1:
            positions = positions[0].unsqueeze(0)
        return super()._reshape_cache(  # pyrefly: ignore [missing-attribute]
            query, positions
        )


class AscComplexRoPE(
    _FirstRowPositionsMixin,
    WorkaroundComplexRoPE,
):
    @dataclass(kw_only=True, slots=True)
    class Config(WorkaroundComplexRoPE.Config):
        pass

    @staticmethod
    def apply_rotary_emb(  # pyrefly: ignore [bad-override]
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rope_cache
        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        if inverse:
            sin = -sin
        xq_out = torch_npu.npu_rotary_mul(query.float(), cos, sin, rotary_mode="interleave").type_as(query)
        if key is None:
            return xq_out
        xk_out = torch_npu.npu_rotary_mul(
            key.float(), cos.to(key.dtype), sin.to(key.dtype), rotary_mode="interleave"
        ).type_as(key)
        return xq_out, xk_out


class AscCosSinRoPE(_FirstRowPositionsMixin, CosSinRoPE):
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
        xq_out = torch_npu.npu_rotary_mul(query.float(), cos, sin, rotary_mode="half").type_as(query)
        xk_out = torch_npu.npu_rotary_mul(
            key.float(), cos.to(key.dtype), sin.to(key.dtype), rotary_mode="half"
        ).type_as(key)
        return xq_out, xk_out


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="AscendC fused ComplexRoPE via torch_npu.npu_rotary_mul (interleave mode)",
)
def asc_complex(cfg: ComplexRoPE.Config) -> AscComplexRoPE.Config:
    return derive(cfg, AscComplexRoPE.Config)


@override(
    target=CosSinRoPE.Config,
    description="AscendC fused CosSinRoPE via torch_npu.npu_rotary_mul (half mode)",
)
def asc_cossin(cfg: CosSinRoPE.Config) -> AscCosSinRoPE.Config:
    return derive(cfg, AscCosSinRoPE.Config)
