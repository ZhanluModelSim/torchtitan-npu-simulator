# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-swap configuration for the V4.1 sparse attention quantization path."""

__all__ = ["DeepSeekV41QuantSparseAttentionConfig"]

from dataclasses import dataclass
from types import MethodType

from torchao.quantization.qat import QATStep
from torchao.quantization.transform_module import register_quantize_module_handler

from torchao_npu.configs.module_swap import ModuleSwapConfig


@dataclass(kw_only=True, slots=True)
class DeepSeekV41QuantSparseAttentionConfig(ModuleSwapConfig):
    """Install FP8 SWA fake quantization; main KV is supplied by QuantCompressor."""


def _quantized_forward(
    self,
    q,
    swa_k,
    attn_sink,
    attention_masks,
    *,
    cmp_k=None,
    topk_indices=None,
    topk_scores=None,
    **kwargs,
):
    from torchao_npu.quantized_modules.deepseek_v4_1.sparse_attention import QuantV41SparseAttention

    # Same argument order as the port this replaces, so the swap stays invisible to the
    # host. Unknown extension keywords remain ignorable, while ``topk_scores`` must keep
    # the SMLAG-to-SLIKG teacher edge intact.
    del kwargs
    return QuantV41SparseAttention.forward(
        self,
        q,
        swa_k,
        attn_sink,
        attention_masks,
        cmp_k=cmp_k,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
    )


@register_quantize_module_handler(DeepSeekV41QuantSparseAttentionConfig)
def _quant_v41_sparse_attention_transform(module, config: DeepSeekV41QuantSparseAttentionConfig):
    if not hasattr(module, "_torchao_npu_original_forward"):
        module._torchao_npu_original_forward = module.forward
    if config.step == QATStep.PREPARE:
        module._torchao_npu_module_swap_config = config
        module.forward = MethodType(_quantized_forward, module)
    else:
        module.forward = module._torchao_npu_original_forward
    return module
