# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.vision.encoder import UnmaskedVisionSDPA


class AscUnmaskedVisionSDPA(UnmaskedVisionSDPA):
    @dataclass(kw_only=True, slots=True)
    class Config(UnmaskedVisionSDPA.Config):
        pass

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # The NPU SDPA backend selector specializes the symbolic patch count.
        # Use the fusion attention interface with symbolic BNSD shapes.
        # Dropout is disabled, so the ordinary interface's
        # Python seed/offset/numels outputs are not needed for mask regeneration.
        return torch_npu.npu_fusion_attention(q, k, v, q.shape[1], "BNSD", scale=q.shape[-1] ** -0.5, keep_prob=1.0)[0]


@override(
    target=UnmaskedVisionSDPA.Config,
    exact=True,
    description="Use fused unmasked vision attention with symbolic patch counts",
)
def asc_unmasked_sdpa(cfg: UnmaskedVisionSDPA.Config) -> AscUnmaskedVisionSDPA.Config:
    return derive(cfg, AscUnmaskedVisionSDPA.Config)
