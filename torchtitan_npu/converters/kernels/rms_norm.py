# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu
from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter

logger = logging.getLogger(__name__)


def _get_eps(module: nn.Module) -> float | None:
    for attr_name in ["eps", "variance_epsilon", "epsilon"]:
        eps = getattr(module, attr_name, None)
        if eps is not None:
            return float(eps)
    return None


class NPURMSNorm(RMSNorm):
    def __init__(self, parent: RMSNorm):
        normalized_shape = parent.normalized_shape
        if len(normalized_shape) != 1:
            raise ValueError(
                "NPURMSNorm supports one-dimensional normalized_shape, got "
                f"{normalized_shape}"
            )
        super().__init__(
            RMSNorm.Config(
                normalized_shape=normalized_shape[0],
                eps=_get_eps(parent),
                elementwise_affine=parent.elementwise_affine,
                param_init=getattr(parent, "_param_init", None),
            )
        )
        if parent.weight is not None:
            self.register_parameter("weight", parent.weight)
        self.eps = _get_eps(parent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matches the default implementation of nn.RMSNorm:
        # - Use user-provided eps if it exists.
        # - Otherwise, use the machine epsilon of the current input `x`.
        resolved_eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return torch_npu.npu_rms_norm(x, self.weight, resolved_eps)[0]

    def _init_self_parameters(self) -> None:
        if getattr(self, "_param_init", None) is not None:
            super()._init_self_parameters()
        else:
            self.reset_parameters()


class NpuRMSNormConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, RMSNorm):
                continue
            replace_module_with_name(model, name, NPURMSNorm(module))


@register_model_converter("npu_rms_norm")
class RMSNormModelConfig(ModelCustomConfig):
    model_converter = NpuRMSNormConverter
