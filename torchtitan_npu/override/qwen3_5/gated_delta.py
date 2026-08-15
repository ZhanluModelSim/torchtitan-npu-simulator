# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.model import GatedDeltaKernel

from torchtitan_npu.ops.triton.gdn import gated_delta_rule as run_gdn


class NPUGatedDeltaKernel(GatedDeltaKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        pass

    def forward(
        self,
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return run_gdn(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN)


@override(target=GatedDeltaKernel.Config, exact=True, description="Use the Triton-Ascend GDN kernel")
def npu_gated_delta(cfg: GatedDeltaKernel.Config) -> NPUGatedDeltaKernel.Config:
    return derive(cfg, NPUGatedDeltaKernel.Config)
