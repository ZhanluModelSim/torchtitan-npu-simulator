# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-swap configuration for the DeepSeek-V4.1 QLI/QSLI path."""

__all__ = ["DeepSeekV41QuantLightningIndexerConfig"]

from dataclasses import dataclass
from types import MethodType

from torchao.quantization.qat import QATStep
from torchao.quantization.transform_module import register_quantize_module_handler

from torchao_npu.configs.module_swap import ModuleSwapConfig


@dataclass(kw_only=True, slots=True)
class DeepSeekV41QuantLightningIndexerConfig(ModuleSwapConfig):
    """Install the ds41 MXFP4 QLI/QSLI selector with its SLIKG backward."""


def _quantized_forward(
    self,
    idx_q_BLHiDi,
    idx_k_BNDi,
    weights_BLHi,
    attention_masks,
    *,
    candidates_BL1C,
):
    from torchao_npu.quantized_modules.deepseek_v4_1.lightning_indexer import QuantV41LightningIndexer

    return QuantV41LightningIndexer.forward(
        self,
        idx_q_BLHiDi,
        idx_k_BNDi,
        weights_BLHi,
        attention_masks,
        candidates_BL1C=candidates_BL1C,
    )


@register_quantize_module_handler(DeepSeekV41QuantLightningIndexerConfig)
def _quant_v41_lightning_indexer_transform(module, config: DeepSeekV41QuantLightningIndexerConfig):
    if not hasattr(module, "_torchao_npu_original_forward"):
        module._torchao_npu_original_forward = module.forward
    if config.step == QATStep.PREPARE:
        module._torchao_npu_module_swap_config = config
        module.forward = MethodType(_quantized_forward, module)
    else:
        module.forward = module._torchao_npu_original_forward
    return module
