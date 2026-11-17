# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 score TopK with a compile-safe TileLang op."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override

from torchtitan_npu.ops.tilelang import tilelang_topk_gate
from torchtitan_npu.patches.torchtitan.models.common.moe import HashRouter


class TilelangHashRouter(HashRouter):
    """DeepSeek-V4 router using TileKernels TopK on non-hash layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(HashRouter.Config):
        pass

    def _select_experts(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        if isinstance(scores_for_choice, DTensor):
            raise ValueError("TileKernels topk_gate does not support DTensor inputs.")
        if scores_for_choice.device.type != "npu":
            raise ValueError("TileKernels topk_gate requires NPU inputs.")
        if scores_for_choice.dtype != torch.float32:
            raise ValueError("TileKernels topk_gate requires float32 scores.")

        num_experts = scores_for_choice.size(-1)
        flat_scores = scores_for_choice.reshape(-1, num_experts).contiguous()
        flat_indices = tilelang_topk_gate(flat_scores, self.top_k)
        return flat_indices.view(*scores_for_choice.shape[:-1], self.top_k)


@override(
    target=HashRouter.Config,
    exact=True,
    description="DeepSeek-V4 non-hash routing via TileKernels topk_gate",
)
def tilelang(cfg: HashRouter.Config) -> TilelangHashRouter.Config:
    return derive(cfg, TilelangHashRouter.Config)
