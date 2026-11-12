# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Upstream-PR: https://github.com/pytorch/torchtitan/pull/5646
#
# [TODO] Short-term backport: the pinned TorchTitan 0.4.0 predates #5646 and does
# not export ``LimitedSwiGLU`` yet. Delete this module and import the upstream
# class once the pinned baseline includes that commit.

"""DeepSeek activation components not yet represented by upstream configs."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.models.common.activation import SwiGLU


class LimitedSwiGLU(SwiGLU):
    """SwiGLU that clamps gate above and up on both sides (DeepSeek-V4 ``swiglu_limit``).

    Upstream #5646 derives this from ``BinaryActivationFn`` so the non-clamping
    fused SwiGLU override does not match it. The NPU fused override
    (``AscSwiGLU``) does honour ``swiglu_limit``, so keep the ``SwiGLU`` base
    until the pinned baseline carries the upstream class.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SwiGLU.Config):
        swiglu_limit: float

        def __post_init__(self) -> None:
            if not math.isfinite(self.swiglu_limit) or self.swiglu_limit <= 0:
                raise ValueError(f"swiglu_limit must be finite and positive, got {self.swiglu_limit}")

    def __init__(self, config: Config) -> None:
        self.swiglu_limit = config.swiglu_limit

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(gate, max=self.swiglu_limit)
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up
