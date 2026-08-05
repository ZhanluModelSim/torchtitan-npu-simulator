# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install the simulator-owned RMSNorm converter without changing production."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.rms_norm import RMSNormModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import SimRMSNorm

if TYPE_CHECKING:
    import torch.nn as nn

_original_rms_norm_converter: type | None = None


class SimRMSNormConverter(ModelCustomConverter):
    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, RMSNorm):
                replace_module_with_name(model, name, SimRMSNorm(module))


def apply_rms_norm_shims() -> None:
    global _original_rms_norm_converter
    if _original_rms_norm_converter is None:
        _original_rms_norm_converter = RMSNormModelConfig.model_converter
    RMSNormModelConfig.model_converter = SimRMSNormConverter


def unapply_rms_norm_shims() -> None:
    global _original_rms_norm_converter
    if _original_rms_norm_converter is not None:
        RMSNormModelConfig.model_converter = _original_rms_norm_converter
        _original_rms_norm_converter = None
