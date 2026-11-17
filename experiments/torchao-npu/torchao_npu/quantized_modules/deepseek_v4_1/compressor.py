# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-RoPE main-KV fake quantization for source compressors."""

from collections.abc import Callable

import torch

from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_compressor import (
    DeepSeekV41QuantCompressorConfig,
)
from torchao_npu.ops.kv_cache_fake_quant import fake_quantize_mx_bf16


class QuantCompressor(torch.nn.Module):
    """Forward behavior installed on the original Compressor instance."""

    _torchao_npu_original_forward: Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]]
    _torchao_npu_module_swap_config: DeepSeekV41QuantCompressorConfig

    def forward(self, x_BLD, positions_BL, cmp_k=None):
        main_kv, latent = self._torchao_npu_original_forward(x_BLD, positions_BL, cmp_k)
        # Reusing layers must return the same shared tensor without another Q/DQ.
        if not self.is_source:
            return main_kv, latent
        if main_kv is None:
            raise RuntimeError("A source compressor must return main KV")
        config = self._torchao_npu_module_swap_config
        return fake_quantize_mx_bf16(main_kv, quant_group_size=config.group_size), latent
