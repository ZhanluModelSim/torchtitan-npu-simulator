# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run RMSNorm with the fused CANN operator."""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm


class CANNRMSNorm(RMSNorm):
    """RMSNorm backed by ``torch_npu.npu_rms_norm``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=RMSNorm.Config,
    description="CANN fused RMSNorm via torch_npu.npu_rms_norm",
)
def cann(cfg: RMSNorm.Config) -> CANNRMSNorm.Config:
    return derive(cfg, CANNRMSNorm.Config)
