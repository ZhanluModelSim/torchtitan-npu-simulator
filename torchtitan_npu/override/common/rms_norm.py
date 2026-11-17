# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run RMSNorm with the fused AscendC operator."""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm


class AscRMSNorm(RMSNorm):
    """RMSNorm backed by ``torch_npu.npu_rms_norm``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not config.elementwise_affine:
            del self.weight
            self.register_buffer("weight", torch.ones(config.normalized_shape), persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # The unit scale is a non-persistent buffer, so FSDP never manages
        # it: under CPU offload it would stay on the host while the fused
        # operator always consumes it with device-resident activations.
        del buffer_device
        if not self.elementwise_affine:
            device = torch_npu.npu.current_device()
            if self.weight.device != device:
                self._buffers["weight"] = self.weight.to(device)
            self.weight.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=RMSNorm.Config,
    description="AscendC fused RMSNorm via torch_npu.npu_rms_norm",
)
def asc(cfg: RMSNorm.Config) -> AscRMSNorm.Config:
    return derive(cfg, AscRMSNorm.Config)
