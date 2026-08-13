"""DeepSeek-V3.2 sparse-attention overrides (registry-facing module).

The CANN implementation (TND metadata extension + fused attention) lives in
``cann.py``; the registrations are defined here so the override paths stay
``override.deepseek_v3_2.sparse_attn.{cann_metadata, cann}``.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.common.metadata_extension import MetadataExtension
from torchtitan_npu.models.deepseek_v3_2.model import SparseInnerAttention

from .cann import (
    CANNSparseIndexerLoss,
    CANNSparseInnerAttention,
    CANNVarlenMetadataExtension,
)


@override(
    target=MetadataExtension.Config,
    description="CANN varlen metadata extension for TND layout (sparse attention).",
)
def cann_metadata(
    cfg: MetadataExtension.Config,
) -> CANNVarlenMetadataExtension.Config:
    return CANNVarlenMetadataExtension.Config(window_size=cfg.window_size)


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
