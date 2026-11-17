# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-swap configuration for main-KV FP4 fake quantization."""

__all__ = ["DeepSeekV41QuantCompressorConfig"]

from dataclasses import dataclass
from types import MethodType

from torchao.quantization.qat import QATStep
from torchao.quantization.transform_module import register_quantize_module_handler

from torchao_npu.configs.module_swap import ModuleSwapConfig


@dataclass(kw_only=True, slots=True)
class DeepSeekV41QuantCompressorConfig(ModuleSwapConfig):
    """Quantize post-RoPE main KV as MXFP4 with BF16 group scales."""

    group_size: int = 16

    def __post_init__(self) -> None:
        ModuleSwapConfig.__post_init__(self)
        if self.group_size not in (16, 32):
            raise ValueError("DeepSeekV41QuantCompressorConfig requires group_size=16 or 32")


@register_quantize_module_handler(DeepSeekV41QuantCompressorConfig)
def _quant_compressor_transform(module, config: DeepSeekV41QuantCompressorConfig):
    from torchao_npu.quantized_modules.deepseek_v4_1.compressor import QuantCompressor

    if not hasattr(module, "_torchao_npu_original_forward"):
        module._torchao_npu_original_forward = module.forward
    if config.step == QATStep.PREPARE:
        module._torchao_npu_module_swap_config = config
        module.forward = MethodType(QuantCompressor.forward, module)
    else:
        module.forward = module._torchao_npu_original_forward
    return module
