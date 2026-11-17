# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common FFN implementation backed by CANN ``SwigluGroup``."""

import importlib
from dataclasses import dataclass

import spmd_types as spmd
import torch

# Keep the PyTorch dispatcher entry used by TorchAO weight wrappers.
from torch import _grouped_mm
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts


def _effective_swiglu_limit(module) -> float | None:
    swiglu_limit = getattr(module, "swiglu_limit", None)
    if swiglu_limit is None or swiglu_limit <= 0:
        return None
    return float(swiglu_limit)


def swiglu_group_activation(
    h: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the documented ``cann_ops_nn.swiglu_group`` op."""
    weight = None if routed_scores is None else routed_scores.to(dtype=torch.float32).contiguous()
    clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
    return torch.ops.cann_ops_nn.swiglu_group.default(
        h.contiguous(),
        weight=weight,
        group_index=None,
        clamp_limit=clamp_limit,
    )


class AscGroupedExperts(GroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    # Match upstream keyword arguments to preserve the GroupedExperts interface.
    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1 = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2 = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3 = self.w3_EFD.to_local()
        else:
            w1 = self.w1_EFD
            w2 = self.w2_EDF
            w3 = self.w3_EFD

        offsets = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets, axis, src=spmd.P, dst=spmd.V)

        gate = _grouped_mm(
            x_RD.bfloat16(),
            w1.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        up = _grouped_mm(
            x_RD.bfloat16(),
            w3.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        hidden = swiglu_group_activation(
            torch.cat((gate, up), dim=-1),
            _effective_swiglu_limit(self),
            routed_scores=routed_scores_R,
        )
        return _grouped_mm(
            hidden,
            w2.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(x_RD)


class AscFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
        hidden = swiglu_group_activation(packed, _effective_swiglu_limit(self))
        return self.w2(hidden)


def _ensure_cann_ops_loaded() -> None:
    importlib.import_module("cann_ops_nn.ops")
