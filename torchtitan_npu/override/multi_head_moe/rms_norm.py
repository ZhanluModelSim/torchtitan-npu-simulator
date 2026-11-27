# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: multi_head_moe RMSNorm on the fused AscendC operator.

The reference ``multi_head_moe`` ships its own local ``RMSNorm`` (reference
``x * rsqrt(mean(x^2) + eps) * weight`` in fp32, plus an explicit fp32 upcast
and cast-back per call). The class is a tree-declared ``Configurable``, so the
shared ``override.common.rms_norm.asc`` — which targets torchtitan's common
``RMSNorm`` — cannot reach it. ``torch_npu.npu_rms_norm`` performs the same
fp32-internal math in one kernel (the system3 ``AscSystem3RMSNorm`` precedent).
"""

from dataclasses import dataclass

import torch
import torch_npu

from torchtitan.config import derive, override

from torchtitan_npu.models.multi_head_moe.model import RMSNorm


class AscMHMRMSNorm(RMSNorm):
    """multi_head_moe RMSNorm backed by ``torch_npu.npu_rms_norm``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=RMSNorm.Config,
    description="AscendC fused multi_head_moe RMSNorm via torch_npu.npu_rms_norm",
)
def asc(cfg: RMSNorm.Config) -> AscMHMRMSNorm.Config:
    return derive(cfg, AscMHMRMSNorm.Config)
