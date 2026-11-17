# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public quantization configurations for torchao-npu."""

from torchao_npu.configs.module_swap import ModuleSwapConfig
from torchao_npu.configs.module_swap_configs import (
    DeepSeekV4QuantLightningIndexerConfig,
    DeepSeekV41QuantCompressorConfig,
    DeepSeekV41QuantLightningIndexerConfig,
    DeepSeekV41QuantSparseAttentionConfig,
    quant_mode_for_config,
)
from torchao_npu.configs.param_swap import ParamSwapConfig

__all__ = [
    "DeepSeekV4QuantLightningIndexerConfig",
    "DeepSeekV41QuantCompressorConfig",
    "DeepSeekV41QuantLightningIndexerConfig",
    "DeepSeekV41QuantSparseAttentionConfig",
    "ModuleSwapConfig",
    "ParamSwapConfig",
    "quant_mode_for_config",
]
