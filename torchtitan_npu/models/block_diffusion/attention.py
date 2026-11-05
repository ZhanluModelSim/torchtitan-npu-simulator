# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native exact block visibility and explicitly selected legacy compute proxy.

This module manually implements the raw workload contract; it does not import
or rewrite the raw reference. CPU, NPU and simulator implementations share the
same sequence segments so converters cannot silently change visibility.
"""

import math
from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import ScaledDotProductAttention


def attention_segments(
    sequence_length: int,
    *,
    attention_mode: str,
    attention_policy: str,
    block_size: int,
) -> tuple[tuple[int, int, int, bool], ...]:
    """Return (query_start, query_end, key_end, causal) for full-window attention.

    Noncausal rectangular segments implement exact block visibility without a
    materialized quadratic block mask. Their KV lengths must stay physical:
    compute-proxy key-dimension scaling is not applicable to exact segments.
    """
    if sequence_length <= 0:
        raise ValueError("attention sequence length must be positive")
    if attention_mode == "compute_proxy":
        return ((0, sequence_length, sequence_length, True),)
    if attention_mode != "exact":
        raise ValueError("attention_mode must be exact or compute_proxy")
    if block_size <= 0 or sequence_length < block_size or sequence_length % block_size:
        raise ValueError("exact attention requires a positive multiple of block_size")
    if attention_policy == "block_causal":
        return tuple(
            (start, start + block_size, start + block_size, False) for start in range(0, sequence_length, block_size)
        )
    if attention_policy == "prefix_causal_canvas_bidirectional":
        prefix = sequence_length - block_size
        segments = ((0, prefix, prefix, True),) if prefix else ()
        return (*segments, (prefix, sequence_length, sequence_length, False))
    raise ValueError("unsupported exact attention_policy")


class ScaledCausalSDPA(ScaledDotProductAttention):
    """Block attention with an opt-in exact path; name retained for compatibility.

    ``compute_proxy`` preserves the original full causal forward and simulator
    FLOP scaling. ``exact`` implements raw contract visibility with alpha=1 and
    never shortens a key dimension. Context-parallel sequence sharding is not
    supported by the exact path until global positions/ownership are modeled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductAttention.Config):
        compute_alpha: float = 1.0
        attention_mode: str = "compute_proxy"
        attention_policy: str = "block_causal"
        block_size: int = 256

        def __post_init__(self) -> None:
            if not math.isfinite(self.compute_alpha) or self.compute_alpha <= 0:
                raise ValueError("compute_alpha must be finite and greater than zero")
            if self.attention_mode not in ("exact", "compute_proxy"):
                raise ValueError("attention_mode must be exact or compute_proxy")
            if self.attention_policy not in ("block_causal", "prefix_causal_canvas_bidirectional"):
                raise ValueError("unsupported attention_policy")
            if type(self.block_size) is not int or self.block_size <= 0:
                raise ValueError("block_size must be a positive integer")
            if self.attention_mode == "exact" and self.compute_alpha != 1.0:
                raise ValueError("exact attention requires compute_alpha=1")

    def __init__(self, config: Config) -> None:
        config.__post_init__()
        super().__init__(config)
        self.compute_alpha = float(config.compute_alpha)
        self.attention_mode = config.attention_mode
        self.attention_policy = config.attention_policy
        self.block_size = config.block_size

    def converter_config(self) -> Config:
        """Copy the complete semantic contract when a converter replaces us."""
        return ScaledCausalSDPA.Config(
            compute_alpha=self.compute_alpha,
            attention_mode=self.attention_mode,
            attention_policy=self.attention_policy,
            block_size=self.block_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_masks=None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError("ScaledCausalSDPA does not accept an external attention mask")
        if kwargs:
            raise TypeError(f"unsupported attention keyword arguments: {sorted(kwargs)}")
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("ScaledCausalSDPA requires equal full-window Q/K/V sequence lengths")
        if self.attention_mode == "compute_proxy":
            return super().forward(q, k, v, scale=scale, enable_gqa=enable_gqa, is_causal=True)
        segments = attention_segments(
            q.shape[1],
            attention_mode=self.attention_mode,
            attention_policy=self.attention_policy,
            block_size=self.block_size,
        )
        outputs = [
            super(ScaledCausalSDPA, self).forward(
                q[:, start:end],
                k[:, :key_end],
                v[:, :key_end],
                scale=scale,
                enable_gqa=enable_gqa,
                is_causal=causal,
            )
            for start, end, key_end, causal in segments
        ]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
