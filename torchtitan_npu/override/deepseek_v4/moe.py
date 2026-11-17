# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 SwiGLU overrides backed by the TileLang custom-op module.

The compile-safe custom operators live in ``torchtitan_npu.ops.tilelang``;
this module keeps only the TorchTitan component replacements and configuration
override factories.
"""

from dataclasses import dataclass
from typing import cast

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts

from torchtitan_npu.ops.tilelang.swiglu import tilelang_swiglu as _run_swiglu


class TilelangGroupedExperts(GroupedExperts):
    """Grouped routed experts with a TileLang SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(x_RD, DTensor):
            raise ValueError(
                "TileLang SwiGLU does not support DTensor expert inputs; use the validated single-rank path"
            )
        if any(isinstance(weight, DTensor) for weight in (self.w1_EFD, self.w2_EDF, self.w3_EFD)):
            raise ValueError(
                "TileLang SwiGLU does not support DTensor expert weights; use the validated single-rank path"
            )

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        gate_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=self.w1_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        up_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=self.w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        swiglu_limit = cast("float", self.swiglu_limit)
        limit = swiglu_limit if swiglu_limit > 0 else None
        h_RF = _run_swiglu(gate_RF, up_RF, limit)
        if routed_scores_R is not None:
            h_RF = (h_RF.float() * routed_scores_R.float().reshape(-1, 1)).to(h_RF.dtype)
        return self._grouped_mm(
            A=h_RF,
            B_t=self.w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)


class TilelangFeedForward(FeedForward):
    """Shared experts with a TileLang SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, DTensor):
            raise ValueError(
                "TileLang SwiGLU does not support DTensor shared-expert inputs; use the validated single-rank path"
            )
        if any(isinstance(weight, DTensor) for weight in (self.w1.weight, self.w2.weight, self.w3.weight)):
            raise ValueError(
                "TileLang SwiGLU does not support DTensor shared-expert weights; use the validated single-rank path"
            )

        gate = self.w1(x)
        up = self.w3(x)
        gate_shape = gate.shape
        gate = gate.reshape(-1, gate.shape[-1]).contiguous()
        up = up.reshape(-1, up.shape[-1]).contiguous()
        swiglu_limit = cast("float", self.swiglu_limit)
        limit = swiglu_limit if swiglu_limit > 0 else None
        hidden = _run_swiglu(gate, up, limit).reshape(gate_shape)
        return self.w2(hidden)


@override(
    target=GroupedExperts.Config,
    exact=True,
    description="DeepSeek-V4 routed experts with TileLang SwiGLU forward/backward",
)
def tilelang_swiglu_grouped(cfg: GroupedExperts.Config) -> TilelangGroupedExperts.Config:
    return derive(cfg, TilelangGroupedExperts.Config)


@override(
    target=FeedForward.Config,
    exact=True,
    description="DeepSeek-V4 shared experts with TileLang SwiGLU forward/backward",
)
def tilelang_swiglu_shared(cfg: FeedForward.Config) -> TilelangFeedForward.Config:
    return derive(cfg, TilelangFeedForward.Config)
