# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pieces for the GLM-5.3-Flash package: the sigmoid-gated RMSNorm
and the init helpers mirroring the production ``_init_weights``.

Note the GLM ``RMSNormGated`` applies a SIGMOID gate, unlike the
silu-gated variant used by qwen3_5/kimi_k3.
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch_npu
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

WEIGHT_INIT = partial(nn.init.normal_, mean=0.0, std=0.02)
LINEAR_INIT = {"weight": WEIGHT_INIT}


def linear_config(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init=LINEAR_INIT,
    )


def kda_dt_bias_init(param: torch.Tensor) -> None:
    """FLA dt_bias recipe: (stable) inverse softplus of a clamped uniform."""
    with torch.no_grad():
        nn.init.uniform_(param, a=math.log(1e-3), b=math.log(1e-1))
        dt = param.exp().clamp_min(1e-4)
        param.copy_(dt + torch.log(-torch.expm1(-dt)))


class RMSNormGated(Module):
    """RMSNorm gated by a SIGMOID activation: norm(x) * sigmoid(gate).

    Mirrors ``Glm5NextTextRMSNormGated`` (strict fp32 norm + fp32 gate)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        eps: float = 1e-6

        def __post_init__(self) -> None:
            self.param_init = {"weight": nn.init.ones_}

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = torch_npu.npu_rms_norm(x, self.weight.to(torch.float32), self.eps)[0]
        return (x * torch.sigmoid(gate.float())).to(input_dtype)
