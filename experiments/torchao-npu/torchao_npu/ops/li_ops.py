# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized QuantLightningIndexerV2 operator bridge."""

__all__ = [
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "quantize_lightning_indexer_input",
]

import cann_ops_transformer
import torch

from torchao_npu.quantization.quant_configs import FP8QuantizeConfig, HiF8QuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.fp8 import fp8_quantize
from torchao_npu.quantization.quant_primitives.hif8 import quantize_hifloat8
from torchao_npu.quantization.quant_primitives.mx import mx_quantize


def quantize_lightning_indexer_input(value: torch.Tensor, config):
    """Quantize one LI Q/K input according to its TorchAO config."""

    if isinstance(config, FP8QuantizeConfig):
        return fp8_quantize(value, config)
    if isinstance(config, MXQuantizeConfig):
        q_data, descale = mx_quantize(value, -1, config)
        # The LI quant_mode identifies packed FP4; CANN consumes uint8 storage.
        if config.elem_dtype is torch.float4_e2m1fn_x2:
            q_data = q_data.view(torch.uint8)
        return q_data, descale.view(torch.float8_e8m0fnu)
    if isinstance(config, HiF8QuantizeConfig):
        return quantize_hifloat8(value, config)
    raise TypeError(f"Unsupported QuantLightningIndexer quantization config: {type(config).__name__}")


def quant_lightning_indexer_metadata(
    *,
    index_n_heads: int,
    index_head_dim: int,
    topk: int,
    quant_mode: int,
    layout_q: str,
    layout_k: str,
    mask_mode: int,
    cmp_ratio: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cmp_residual_k: torch.Tensor,
):
    """Build metadata paired with ``quant_lightning_indexer``."""

    return cann_ops_transformer.quant_lightning_indexer_metadata(
        index_n_heads,
        1,
        index_head_dim,
        topk,
        quant_mode,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
    )


def quant_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    q_config,
    k_config,
    *,
    topk: int,
    quant_mode: int,
    layout_q: str,
    layout_k: str,
    mask_mode: int,
    cmp_ratio: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cmp_residual_k: torch.Tensor,
    metadata: torch.Tensor,
):
    """Run QuantLightningIndexerV2 on packed TND quantized Q/K tensors."""

    q_data, q_descale = quantize_lightning_indexer_input(q, q_config)
    k_data, k_descale = quantize_lightning_indexer_input(k, k_config)

    return cann_ops_transformer.quant_lightning_indexer(
        q_data,
        k_data,
        w,
        q_descale,
        k_descale,
        topk,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=0,
    )
