from dataclasses import dataclass

import torch
import torch_npu

from torchtitan.config import derive, override
from torchtitan.models.common.rope import (
    ComplexRoPE,
    CosSinRoPE,
    _maybe_check_max_pos,
    _maybe_wrap_positions,
    _reshape_for_broadcast,
)


class WorkaroundComplexRoPE(ComplexRoPE):
    """Workaround for ComplexRoPE — indexes via real representation."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        cache_real = torch.view_as_real(self.cache)
        cache_flat = cache_real.flatten(-2, -1)
        reshaped = _reshape_for_broadcast(cache_flat, complex_query_shape, positions)
        reshaped = reshaped.view(*reshaped.shape[:-1], -1, 2)
        return torch.view_as_complex(reshaped)


class NPUComplexRoPE(WorkaroundComplexRoPE):
    """NPU fused RoPE using ``torch_npu.npu_rotary_mul``.

    Groups the real/imag into cos/sin arrays and delegates to a single
    ``npu_rotary_mul`` call in ``interleave`` mode instead of a manual
    complex multiply.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(WorkaroundComplexRoPE.Config):
        pass

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The fused NPU interleaved RoPE op aclnnRotaryPositionEmbeddingV2
        only supports cos/sin broadcast over the batch dim, i.e. positions
        shared across batch.
        """
        if positions is not None and positions.size(0) > 1:
            positions = positions[0].unsqueeze(0)
        return super()._reshape_cache(query, positions)

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


class NPUCosSinRoPE(CosSinRoPE):
    """NPU fused CosSinRoPE using ``torch_npu.npu_rotary_mul`` in ``half`` mode."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The fused NPU interleaved RoPE op aclnnRotaryPositionEmbeddingV2
        only supports cos/sin broadcast over the batch dim, i.e. positions
        shared across batch.
        """
        if positions is not None and positions.size(0) > 1:
            positions = positions[0].unsqueeze(0)
        return super()._reshape_cache(query, positions)

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
    description="NPU ComplexRoPE workaround (indexes via view_as_real float32)",
)
def complex_workaround(cfg: ComplexRoPE.Config) -> WorkaroundComplexRoPE.Config:
    return derive(cfg, WorkaroundComplexRoPE.Config)


@override(
    target=ComplexRoPE.Config,
    description="NPU fused RoPE via torch_npu.npu_rotary_mul (interleave mode)",
)
def complex_fused(cfg: ComplexRoPE.Config) -> NPUComplexRoPE.Config:
    return derive(cfg, NPUComplexRoPE.Config)


@override(
    target=CosSinRoPE.Config,
    description="NPU fused CosSinRoPE via torch_npu.npu_rotary_mul (half mode)",
)
def cossin_fused(cfg: CosSinRoPE.Config) -> NPUCosSinRoPE.Config:
    return derive(cfg, NPUCosSinRoPE.Config)
