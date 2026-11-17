# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete module-swap quantization configs."""

__all__ = [
    "DeepSeekV4QuantLightningIndexerConfig",
    "DeepSeekV41QuantCompressorConfig",
    "DeepSeekV41QuantLightningIndexerConfig",
    "DeepSeekV41QuantSparseAttentionConfig",
    "quant_mode_for_config",
]

from torchao_npu.configs.module_swap_configs.deepseek_v4.quant_lightning_indexer import (
    DeepSeekV4QuantLightningIndexerConfig,
    quant_mode_for_config,
)
from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_compressor import (
    DeepSeekV41QuantCompressorConfig,
)
from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_lightning_indexer import (
    DeepSeekV41QuantLightningIndexerConfig,
)
from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_sparse_attention import (
    DeepSeekV41QuantSparseAttentionConfig,
)
