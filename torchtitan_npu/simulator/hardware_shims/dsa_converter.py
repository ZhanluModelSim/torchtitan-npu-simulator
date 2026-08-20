# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install the simulator-owned DeepSeek V3.2 DSA converter."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

from torchtitan_npu.converters.kernels.dsa import DSAModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.deepseek_v32.model import DSV32_SDPA
from torchtitan_npu.simulator.hardware_shims.dsa_shim import sim_dsa_forward

if TYPE_CHECKING:
    import torch.nn as nn

_original_dsa_converter: type | None = None
_DSA_SHIM_MARKER = "_simulator_dsa_shim_installed"


class SimDSAConverter(ModelCustomConverter):
    """Bind shape-only DSA methods without replacing parallelized modules."""

    def convert(self, model: nn.Module) -> None:
        if self.model_name != "deepseek_v32":
            return
        for module in model.modules():
            if isinstance(module, DSV32_SDPA) and not getattr(
                module,
                _DSA_SHIM_MARKER,
                False,
            ):
                module.forward = MethodType(sim_dsa_forward, module)
                setattr(module, _DSA_SHIM_MARKER, True)


def apply_dsa_shims() -> None:
    global _original_dsa_converter
    if _original_dsa_converter is None:
        _original_dsa_converter = DSAModelConfig.model_converter
    DSAModelConfig.model_converter = SimDSAConverter


def unapply_dsa_shims() -> None:
    global _original_dsa_converter
    if _original_dsa_converter is not None:
        DSAModelConfig.model_converter = _original_dsa_converter
        _original_dsa_converter = None
