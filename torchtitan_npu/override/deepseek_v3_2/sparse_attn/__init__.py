"""DeepSeek-V3.2 sparse-attention overrides (registry-facing module).

The CANN implementation (TND mask handler + fused attention) lives in
``cann.py``; the registrations are defined here so the override paths stay
``override.deepseek_v3_2.sparse_attn.{cann_metadata, cann}``.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v3_2.model import SparseInnerAttention
from torchtitan_npu.patches.torchtitan.models.common.mask_handler import BaseMaskHandler

from .cann import (
    CANNSparseIndexerLoss,
    CANNSparseInnerAttention,
    CANNVarlenMetadataHandler,
)


@override(
    target=BaseMaskHandler.Config,
    description="CANN varlen metadata handler for TND layout (sparse attention).",
)
def cann_metadata(
    cfg: BaseMaskHandler.Config,
) -> CANNVarlenMetadataHandler.Config:
    return CANNVarlenMetadataHandler.Config()


@override(
    target=SparseInnerAttention.Config,
    description="CANN sparse flash attention via torch_npu.npu_sparse_flash_attention",
)
def cann(
    cfg: SparseInnerAttention.Config,
) -> CANNSparseInnerAttention.Config:
    return CANNSparseInnerAttention.Config(
        sharding_config=cfg.sharding_config,
        index_topk=cfg.index_topk,
        indexer_loss=derive(cfg.indexer_loss, CANNSparseIndexerLoss.Config),
    )
