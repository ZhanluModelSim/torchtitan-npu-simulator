# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi SiTU experts on native TorchTitan routing and dispatch/combine."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module


class SituGLU(Module):
    """Official SiTU gate and optional bounded up branch, evaluated in FP32."""

    def __init__(self, beta: float = 4.0, linear_beta: float | None = 25.0):
        super().__init__()
        if beta <= 0 or (linear_beta is not None and linear_beta <= 0):
            raise ValueError("SiTU beta and optional linear_beta must be positive")
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        activated_gate = self.beta * torch.tanh(gate.float() / self.beta) * torch.sigmoid(gate.float())
        up = up.float()
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (activated_gate * up).to(x.dtype)


class KimiMLP(FeedForward):
    """Native dense/shared projections, with Kimi's SiTU activation boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        beta: float = 4.0
        linear_beta: float | None = 25.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.act_fn = SituGLU(config.beta, config.linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(torch.cat((self.w1(x), self.w3(x)), dim=-1)))


class KimiGroupedExperts(GroupedExperts):
    """Native expert layout and grouped-MM seam; only activation differs."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        beta: float = 4.0
        linear_beta: float | None = 25.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.act_fn = SituGLU(config.beta, config.linear_beta)

    def forward(
        self, x_RD: torch.Tensor, num_tokens_per_expert_E: torch.Tensor, *, routed_scores_R: torch.Tensor | None = None
    ) -> torch.Tensor:
        if routed_scores_R is not None:
            raise ValueError("Kimi preserves post-down-projection router weighting in the native dispatcher")
        weights = (self.w1_EFD, self.w2_EDF, self.w3_EFD)
        w1, w2, w3 = tuple(w.to_local() if isinstance(w, DTensor) else w for w in weights)
        offsets = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate = self._grouped_mm(A=x_RD.bfloat16(), B_t=w1.bfloat16().transpose(-2, -1), offs=offsets)
        up = self._grouped_mm(A=x_RD.bfloat16(), B_t=w3.bfloat16().transpose(-2, -1), offs=offsets)
        hidden = self.act_fn(torch.cat((gate, up), dim=-1))
        return self._grouped_mm(A=hidden, B_t=w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x_RD)


class KimiLatentMoE(MoE):
    """Native router, routed experts and shared branch with a latent bottleneck.

    The bottleneck surrounds dispatch/combine, so normalization observes the
    combined output (including any TP reduction), never individual experts.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        down_proj: Linear.Config | None = None
        up_proj: Linear.Config | None = None
        norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        if (config.down_proj is None) != (config.up_proj is None):
            raise ValueError("Latent MoE requires both down and up projections")
        self.down_proj = config.down_proj.build() if config.down_proj is not None else None
        self.up_proj = config.up_proj.build() if config.up_proj is not None else None
        self.norm = config.norm.build() if config.norm is not None else None

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        scores, expert_ids, full_scores = self.router(x_BLD, self.expert_bias_E)
        counts = torch.zeros_like(full_scores, dtype=torch.bool).scatter_(-1, expert_ids, True).sum((0, 1))
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(counts)
        latent = self.down_proj(x_BLD) if self.down_proj is not None else x_BLD
        combined = self.routed_experts(latent, scores, expert_ids, counts)
        if self.norm is not None:
            combined = self.norm(combined)
        output = self.up_proj(combined) if self.up_proj is not None else combined
        if self.shared_experts is not None:
            output = output + self.shared_experts(x_BLD)
        return output
