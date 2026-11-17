# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU QuantLightningIndexer module behavior."""

__all__ = [
    "QuantizedLightningIndexerInputs",
    "quantized_lightning_indexer_forward",
]

from dataclasses import dataclass

import torch

from torchao_npu.ops.li_ops import quant_lightning_indexer


@dataclass(frozen=True, slots=True)
class QuantizedLightningIndexerInputs:
    """Host-normalized inputs consumed by the quantized LI kernel."""

    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    topk: int
    metadata: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cmp_residual_k: torch.Tensor


def quantized_lightning_indexer_forward(self, idx_q, idx_k, idx_w, *, attention_masks):
    """Run the quantized LI kernel using a host-provided input adapter."""

    config = self._torchao_npu_module_swap_config
    if config.input_adapter is None:
        raise RuntimeError("Quantized LI requires a host input adapter")
    inputs = config.input_adapter(self, idx_q, idx_k, idx_w, attention_masks)
    sparse_indices, _ = quant_lightning_indexer(
        inputs.query,
        inputs.key,
        inputs.weights,
        config.query_config,
        config.key_config,
        topk=inputs.topk,
        quant_mode=config.quant_mode,
        layout_q=config.layout_q,
        layout_k=config.layout_k,
        mask_mode=config.mask_mode,
        cmp_ratio=config.cmp_ratio,
        cu_seqlens_q=inputs.cu_seqlens_q,
        cu_seqlens_k=inputs.cu_seqlens_k,
        cmp_residual_k=inputs.cmp_residual_k,
        metadata=inputs.metadata,
    )
    return sparse_indices.reshape(idx_q.shape[0], idx_q.shape[1], 1, -1)
