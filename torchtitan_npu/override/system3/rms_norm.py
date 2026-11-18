# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: system3 RMSNorm on the fused AscendC operator.

The reference ``System3RMSNorm`` normalizes in float32 and casts back to the
input dtype; ``torch_npu.npu_rms_norm`` performs the same fp32-internal math
in one kernel.  The weight stays replicated, so the DTensor dispatch feeds
the local shard to the vendor op exactly like the common ``AscRMSNorm``.
"""

import torch
import torch_npu
from dataclasses import dataclass

from torchtitan.config import derive, override

from torchtitan_npu.models.system3.model import System3RMSNorm


class AscSystem3RMSNorm(System3RMSNorm):
    """RMSNorm backed by ``torch_npu.npu_rms_norm``."""

    @dataclass(kw_only=True, slots=True)
    class Config(System3RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=System3RMSNorm.Config,
    description="AscendC fused RMSNorm via torch_npu.npu_rms_norm",
)
def asc(cfg: System3RMSNorm.Config) -> AscSystem3RMSNorm.Config:
    return derive(cfg, AscSystem3RMSNorm.Config)
