# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU QuantLightningIndexer metadata implementation."""

__all__ = [
    "QuantizedLightningIndexerMetadata",
    "QuantizedLightningIndexerMetadataConfig",
]

from dataclasses import dataclass

import torch

from torchao_npu.ops.li_ops import quant_lightning_indexer_metadata


@dataclass(kw_only=True, slots=True)
class QuantizedLightningIndexerMetadataConfig:
    """Quantized LI metadata parameters supplied by the host integration."""

    index_n_heads: int
    index_head_dim: int
    index_topk: int
    layout_q: str
    layout_k: str
    mask_mode: int
    cmp_ratio: int
    quant_mode: int = 5


class QuantizedLightningIndexerMetadata:
    """Build quantized LI metadata from explicit varlen kernel inputs.

    The host integration owns its metadata container and plan mutation. This
    class only owns the quantized kernel call and has no dependency on host
    model or metadata container types.
    """

    Config = QuantizedLightningIndexerMetadataConfig

    def __init__(self, config: QuantizedLightningIndexerMetadataConfig):
        self.config = config

    def __call__(
        self,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        cmp_residual_k: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        return quant_lightning_indexer_metadata(
            index_n_heads=cfg.index_n_heads,
            index_head_dim=cfg.index_head_dim,
            topk=cfg.index_topk,
            quant_mode=cfg.quant_mode,
            layout_q=cfg.layout_q,
            layout_k=cfg.layout_k,
            mask_mode=cfg.mask_mode,
            cmp_ratio=cfg.cmp_ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
        )
