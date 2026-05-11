# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import ClassVar

import torch.nn as nn

import torch_npu
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from ..patches.quantization.quant_config import (
    MoETrainingConfig as TorchMoETrainingConfig,
    MXLinearConfig as TorchMXLinearConfig,
)
from ..patches.quantization.quantize import grouped_quantize_, linear_quantize_


@dataclass(kw_only=True, slots=True)
class NPUMXFP8Config(QuantizationConverter.Config):
    _quantization_type: ClassVar[str] = "mxfp8"
    recipe_name: str = "mxfp8"
    fqns: list[str] = field(default_factory=list)
    filter_fqns: list[str] = field(default_factory=list)


def is_a5():
    try:
        device_name = torch_npu.npu.get_device_name()
        return "Ascend950" in device_name or "Ascend910_95" in device_name
    except Exception:
        return False


def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    return not any(filter_fqn in fqn for filter_fqn in filter_fqns)


def npu_quant_mxfp8_converter_init(
    self,
    config: NPUMXFP8Config,
    *,
    parallel_dims: ParallelDims,
    model_compile_enabled: bool,
):
    self.enabled = False
    if not is_a5():
        raise RuntimeError(
            "[MXFP8/Hif8] is only supported on Ascend950 or higher architecture."
        )

    self.linear_config = TorchMXLinearConfig.from_recipe_name(config.recipe_name)
    self.grouped_mm_config = TorchMoETrainingConfig.from_recipe_name(config.recipe_name)
    self.filter_fqns = config.filter_fqns
    self.moe_fqns = config.fqns
    self.recipe_name = config.recipe_name
    self.pad_token_groups_for_grouped_mm = not parallel_dims.ep_enabled
    self.enabled = True
    logger.info(f"MX training active with recipe {config.recipe_name}")


def npu_quant_mxfp8_converter(self, model: nn.Module):
    if not self.enabled:
        return

    linear_quantize_(
        model,
        config=self.linear_config,
        filter_fn=lambda mod, fqn: module_filter_fn(mod, fqn, self.filter_fqns),
    )
    logger.info("Swapped to MXLinear_NPU layers")

    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in self.moe_fqns:
            if target_fqn in cur_fqn:
                return True
        return False

    grouped_quantize_(
        model,
        config=self.grouped_mm_config,
        filter_fn=moe_module_filter_fn,
    )
    logger.info(
        f"Converted all MoE grouped MM layers "
        f"to use dynamic {self.recipe_name} quantization with scaled grouped GEMMs"
    )


MXFP8Converter.Config = NPUMXFP8Config
MXFP8Converter.__init__ = npu_quant_mxfp8_converter_init
MXFP8Converter.convert = npu_quant_mxfp8_converter
