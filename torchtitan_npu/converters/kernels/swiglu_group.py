# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Select ``cann_ops_nn.swiglu_group`` for routed and shared experts.

Routed experts automatically reuse the model and checkpoint path provided by ``npu_gmm``.
Shared experts remain independently convertible through ``NpuSharedExperts``.
The operator's Autograd integration is provided by ``cann_ops_nn``.
"""

import importlib

import torch
from torch import nn
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts

from torchtitan_npu.converters.kernels import gmm
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.common.moe import NpuSharedExperts


def swiglu_group_activation(
    h: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the documented SwigluGroup op with expert activation inputs."""
    weight = None if routed_scores is None else routed_scores.to(dtype=torch.float32).contiguous()
    clamp_limit = -1.0 if swiglu_limit is None else float(swiglu_limit)
    return torch.ops.cann_ops_nn.swiglu_group.default(
        h.contiguous(),
        weight=weight,
        group_index=None,
        clamp_limit=clamp_limit,
    )


class NpuSwigluGroupConverter(ModelCustomConverter):
    """Select the SwigluGroup activation for routed and shared experts."""

    def convert(self, model: nn.Module) -> None:
        routed_experts: list[gmm.NpuGroupedExperts] = []
        unconverted_routed_experts: list[GroupedExperts] = []
        for module in model.modules():
            if isinstance(module, gmm.NpuGroupedExperts):
                routed_experts.append(module)
            elif isinstance(module, GroupedExperts):
                unconverted_routed_experts.append(module)

        shared_experts = [
            module
            for name, module in model.named_modules()
            if name.rsplit(".", 1)[-1] == "shared_experts" and isinstance(module, FeedForward)
        ]
        if not routed_experts and not unconverted_routed_experts and not shared_experts:
            return

        importlib.import_module("cann_ops_nn.ops")

        if unconverted_routed_experts:
            gmm.NpuGroupedExpertConverter(self.model_spec).convert(model)
            routed_experts = [module for module in model.modules() if isinstance(module, gmm.NpuGroupedExperts)]

        for module in routed_experts:
            module.set_expert_activation(swiglu_group_activation)

        for shared_module in shared_experts:
            NpuSharedExperts.convert(
                shared_module,
                activation_fn=swiglu_group_activation,
            )


@register_model_converter("npu_swiglu_group")
class SwigluGroupModelConfig(ModelCustomConfig):
    model_converter = NpuSwigluGroupConverter
    state_dict_updater = gmm.GMMStateDictUpdater
    state_dict_updater_predicate = staticmethod(gmm.has_npu_grouped_experts)
