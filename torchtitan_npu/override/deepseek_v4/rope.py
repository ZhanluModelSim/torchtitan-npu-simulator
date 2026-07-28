# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: apply DeepSeek-V4 YaRN frequencies to NPU rotary embeddings.

Compatible and fused variants are provided; select one per RoPE config.
"""

import math
from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan_npu.override.common.rope import (
    NPUComplexRoPE,
    NPUFusedRoPE,
    NPUFusedSingleRoPE,
    NPUSingleComplexRoPE,
)
from torchtitan_npu.patches.torchtitan.models.common.rope import SingleComplexRoPE


def precompute_complex_cache_dsv4_yarn(cfg) -> torch.Tensor:
    """Build the complex cache with the DeepSeek-V4 YaRN policy."""
    dim = cfg.dim
    end = cfg.max_seq_len
    theta = cfg.theta
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if cfg.scaling == "yarn" and cfg.original_seq_len > 0:
        base = theta
        original_seq_len = cfg.original_seq_len
        factor = cfg.rope_factor

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return (
                dim
                * math.log(max_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
                max_val - min_val
            )
            return torch.clamp(linear_func, 0, 1)

        low, high = find_correction_range(
            cfg.beta_fast, cfg.beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


class DSV4ComplexRoPE(NPUComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUComplexRoPE.Config):
        pass

    def _precompute_cache(self) -> torch.Tensor:
        return precompute_complex_cache_dsv4_yarn(self.config)


class DSV4SingleComplexRoPE(NPUSingleComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUSingleComplexRoPE.Config):
        pass

    def _precompute_cache(self) -> torch.Tensor:
        return precompute_complex_cache_dsv4_yarn(self.config)


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="NPU ComplexRoPE with DeepSeek-V4 YaRN frequency policy",
)
def npu_dsv4_rope_override(cfg: ComplexRoPE.Config) -> DSV4ComplexRoPE.Config:
    return derive(cfg, DSV4ComplexRoPE.Config)


@override(
    target=SingleComplexRoPE.Config,
    exact=True,
    description="NPU SingleComplexRoPE with DeepSeek-V4 YaRN frequency policy",
)
def npu_dsv4_single_rope_override(
    cfg: SingleComplexRoPE.Config,
) -> DSV4SingleComplexRoPE.Config:
    return derive(cfg, DSV4SingleComplexRoPE.Config)


class DSV4FusedRoPE(NPUFusedRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUFusedRoPE.Config):
        pass

    def _precompute_cache(self) -> torch.Tensor:
        return precompute_complex_cache_dsv4_yarn(self.config)


class DSV4FusedSingleRoPE(NPUFusedSingleRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUFusedSingleRoPE.Config):
        pass

    def _precompute_cache(self) -> torch.Tensor:
        return precompute_complex_cache_dsv4_yarn(self.config)


@override(
    target=ComplexRoPE.Config,
    exact=True,
    description="NPU fused ComplexRoPE with DeepSeek-V4 YaRN frequency policy",
)
def npu_dsv4_fused_rope_override(
    cfg: ComplexRoPE.Config,
) -> DSV4FusedRoPE.Config:
    return derive(cfg, DSV4FusedRoPE.Config)


@override(
    target=SingleComplexRoPE.Config,
    exact=True,
    description="NPU fused SingleComplexRoPE with DeepSeek-V4 YaRN frequency policy",
)
def npu_dsv4_fused_single_rope_override(
    cfg: SingleComplexRoPE.Config,
) -> DSV4FusedSingleRoPE.Config:
    return derive(cfg, DSV4FusedSingleRoPE.Config)
