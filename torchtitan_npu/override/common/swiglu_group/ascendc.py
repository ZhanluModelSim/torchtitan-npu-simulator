# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common FFN implementation backed by CANN ``SwigluGroup``."""

import importlib
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F

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


def _clamped_swiglu_bounds(module, dim: int):
    """Production clamped-swiglu bounds: the gate half is only clamped from
    above, the up half from both sides (MindSpeed ``_min_bound``/``_max_bound``)."""
    limit = _effective_swiglu_limit(module)
    if limit is None:
        return None, None
    half = dim // 2
    min_bound = torch.cat((torch.full((half,), float("-inf")), torch.full((half,), -limit)))
    max_bound = torch.full((dim,), limit)
    return min_bound, max_bound


def swiglu_group_activation(
    h: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the documented ``cann_ops_nn.swiglu_group`` op."""
    # Lazy load: the override factory also derives on the meta capture path,
    # where the simulator's fake bridge registers these ops instead.
    _ensure_cann_ops_loaded()
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
        # Clamped-swiglu models (GLM-5.3-Flash) carry the limit on the
        # source config; derive copies it so the activation clamps match.
        swiglu_limit: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

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

        # Production recipe (MindSpeed ep dispatcher): one fused gate+up GMM
        # over the concatenated [E, 2I, D] weight, then per-half scalar
        # clamps (gate upper-only, up symmetric) feeding the swiglu op.
        w13 = torch.cat((w1, w3), dim=1)
        gate_up = _grouped_mm(
            x_RD.bfloat16(),
            w13.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        limit = _effective_swiglu_limit(self)
        if limit is not None:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        hidden = swiglu_group_activation(
            torch.cat((gate, up), dim=-1),
            None,
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
        swiglu_limit: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit
        dim = 2 * config.w1.out_features
        min_bound, max_bound = _clamped_swiglu_bounds(config, dim)
        if min_bound is not None:
            self.register_buffer("_min_bound", min_bound, persistent=False)
            self.register_buffer("_max_bound", max_bound, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Production runs one fused gate_up projection; fold the split w1/w3
        # weights per pass (the vision reference does the same fold).
        weight = torch.cat((self.w1.weight, self.w3.weight), dim=0)
        bias = None
        if self.w1.bias is not None:
            bias = torch.cat((self.w1.bias, self.w3.bias), dim=0)
        gate_up = F.linear(x, weight, bias)
        min_bound = getattr(self, "_min_bound", None)
        if min_bound is not None:
            gate_up = torch.clamp(gate_up, min=min_bound.to(gate_up.dtype), max=self._max_bound.to(gate_up.dtype))
        hidden = swiglu_group_activation(gate_up)
        return self.w2(hidden)


def _ensure_cann_ops_loaded() -> None:
    importlib.import_module("cann_ops_nn.ops")
