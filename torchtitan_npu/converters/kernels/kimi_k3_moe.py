# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 NPU converter registration.

Kimi routed experts use ``torch._grouped_mm`` directly. Keeping the converter
entry preserves the grouped layout required by EP/ETP/eFSDP. The converter
also replaces KDA's gated RMSNorm with the fused NPU RMSNorm primitive while
keeping its sigmoid output gate explicit.
"""

import logging

import torch
from torch import nn
import torch_npu

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.kimi_k3.attention import RMSNormGated
from torchtitan_npu.models.kimi_k3.feed_forward import KimiGroupedExperts

logger = logging.getLogger(__name__)


class NpuKimiRMSNormGated(RMSNormGated):
    """Kimi gated RMSNorm backed by ``torch_npu.npu_rms_norm``."""

    def __init__(self, parent: RMSNormGated):
        nn.Module.__init__(self)
        self.eps = parent.eps
        self.register_parameter("weight", parent.weight)

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        normalized = torch_npu.npu_rms_norm(
            x,
            self.weight,
            self.eps,
        )[0]
        return normalized * torch.sigmoid(gate.float()).to(x.dtype)


class NpuKimiK3MoeConverter(ModelCustomConverter):
    """Enable Kimi NPU kernels without changing grouped-expert parameters."""

    def convert(self, model: nn.Module) -> None:
        num_grouped_expert_modules = sum(
            isinstance(module, KimiGroupedExperts)
            for module in model.modules()
        )
        gated_norm_names = [
            name
            for name, module in model.named_modules()
            if name and isinstance(module, RMSNormGated)
        ]
        for name in gated_norm_names:
            module = model.get_submodule(name)
            replace_module_with_name(
                model,
                name,
                NpuKimiRMSNormGated(module),
            )
        logger.info(
            "[npu_kimi_k3_moe] Using native grouped_mm for %d expert modules "
            "and fused RMSNorm for %d KDA gated norms",
            num_grouped_expert_modules,
            len(gated_norm_names),
        )


@register_model_converter("npu_kimi_k3_moe")
class NpuKimiK3MoeModelConfig(ModelCustomConfig):
    model_converter = NpuKimiK3MoeConverter
