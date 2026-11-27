# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified quantized configuration for QuantLightningIndexerV2."""

__all__ = ["DeepSeekV4QuantLightningIndexerConfig", "quant_mode_for_config"]

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType
from typing import Annotated

import torch
import torch_npu
import tyro
from torchao.quantization.qat import QATStep
from torchao.quantization.transform_module import register_quantize_module_handler

from torchao_npu.configs.module_swap import ModuleSwapConfig
from torchao_npu.quantization.quant_configs import FP8QuantizeConfig, HiF8QuantizeConfig, MXQuantizeConfig


def _mxfp4_config() -> MXQuantizeConfig:
    return MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)


_LI_QUANT_MODE_TABLE = (
    (FP8QuantizeConfig, torch.float8_e4m3fn, 1, "pertoken"),
    (MXQuantizeConfig, torch.float8_e4m3fn, 3, None),
    (HiF8QuantizeConfig, torch_npu.hifloat8, 4, "pertensor"),
    (MXQuantizeConfig, torch.float4_e2m1fn_x2, 5, None),
)


def quant_mode_for_config(config: MXQuantizeConfig | HiF8QuantizeConfig | FP8QuantizeConfig) -> int:
    """Return the CANN LI quant mode for one Q/K quantization config."""

    for config_type, elem_dtype, quant_mode, dynamic_quant_mode in _LI_QUANT_MODE_TABLE:
        if isinstance(config, config_type) and config.elem_dtype == elem_dtype:
            if dynamic_quant_mode is not None and config.quant_mode != dynamic_quant_mode:
                raise ValueError(f"LI quant_mode={quant_mode} requires dynamic quant_mode={dynamic_quant_mode!r}")
            return quant_mode
    if isinstance(config, MXQuantizeConfig):
        raise ValueError("DeepSeekV4QuantLightningIndexerConfig does not support this MX element dtype")
    raise TypeError(f"Unsupported LI quantization config: {type(config).__name__}")


@dataclass(kw_only=True)
class DeepSeekV4QuantLightningIndexerConfig(ModuleSwapConfig):
    """Q/K quantization strategy for QuantLightningIndexerV2.

    ``quant_mode`` follows the CANN contract: 1=FP8 per-token-head, 3=MXFP8, 4=HiFloat8, and
    5=MXFP4.  The converter supplies the mode-specific configs and host input
    adapter; standalone users must provide an input adapter explicitly.
    """

    layout_q: str = "TND"
    layout_k: str = "TND"
    mask_mode: int = 3
    cmp_ratio: int = 4
    query_config: MXQuantizeConfig | HiF8QuantizeConfig | FP8QuantizeConfig = field(default_factory=_mxfp4_config)
    key_config: MXQuantizeConfig | HiF8QuantizeConfig | FP8QuantizeConfig = field(default_factory=_mxfp4_config)
    quant_mode: int = 5
    input_adapter: Annotated[Callable | None, tyro.conf.Suppress] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        ModuleSwapConfig.__post_init__(self)
        if type(self.query_config) is not type(self.key_config):
            raise ValueError("query_config and key_config must use the same config type")
        if self.query_config.elem_dtype != self.key_config.elem_dtype:
            raise ValueError("query_config and key_config must use the same element dtype")
        expected_mode = quant_mode_for_config(self.query_config)
        if self.quant_mode != expected_mode:
            raise ValueError(
                f"quant_mode={self.quant_mode} does not match the selected quantization config; "
                f"expected {expected_mode}"
            )


@register_quantize_module_handler(DeepSeekV4QuantLightningIndexerConfig)
def _quant_lightning_indexer_transform(module, config: DeepSeekV4QuantLightningIndexerConfig):
    from torchao_npu.quantized_modules.deepseek_v4.lightning_indexer import quantized_lightning_indexer_forward

    if not hasattr(module, "_torchao_npu_original_forward"):
        module._torchao_npu_original_forward = module.forward
    if config.step == QATStep.PREPARE:
        module._torchao_npu_module_swap_config = config
        module.forward = MethodType(quantized_lightning_indexer_forward, module)
    else:
        module.forward = module._torchao_npu_original_forward
    return module
