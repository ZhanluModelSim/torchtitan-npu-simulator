# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compute-oriented attention primitive for Block Diffusion simulation."""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import ScaledDotProductAttention


class ScaledCausalSDPA(ScaledDotProductAttention):
    """Full-sequence causal attention with simulator-only FLOP scaling.

    ``compute_alpha`` does not alter the numerical forward pass. It is emitted
    as fused-attention metadata and scales only the attention FLOP estimate.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductAttention.Config):
        compute_alpha: float = 1.0

        def __post_init__(self) -> None:
            if self.compute_alpha <= 0:
                raise ValueError("compute_alpha must be greater than zero")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.compute_alpha = float(config.compute_alpha)

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
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("ScaledCausalSDPA requires equal Q/K/V sequence lengths")
        return super().forward(
            q,
            k,
            v,
            scale=scale,
            enable_gqa=enable_gqa,
            is_causal=True,
            **kwargs,
        )
