"""DeepSeek-V3.2 sparse-attention overrides (registry-facing module).

The AscendC implementation (TND metadata extension + fused attention) lives in
``ascendc.py``; the registrations are defined here so the override paths stay
``override.deepseek_v3_2.sparse_attn.{asc_metadata, asc}``.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.common.metadata_extension import MetadataExtension
from torchtitan_npu.models.deepseek_v3_2.model import SparseInnerAttention

from .ascendc import (
    AscSparseIndexerLoss,
    AscSparseInnerAttention,
    AscVarlenMetadataExtension,
)


@override(
    target=MetadataExtension.Config,
    description="AscendC varlen metadata extension for TND layout (sparse attention).",
)
def asc_metadata(
    cfg: MetadataExtension.Config,
) -> AscVarlenMetadataExtension.Config:
    return AscVarlenMetadataExtension.Config(window_size=cfg.window_size)


@override(
    target=SparseInnerAttention.Config,
    description="AscendC sparse flash attention via torch_npu.npu_sparse_flash_attention",
)
def asc(
    cfg: SparseInnerAttention.Config,
) -> AscSparseInnerAttention.Config:
    return AscSparseInnerAttention.Config(
        sharding_config=cfg.sharding_config,
        index_topk=cfg.index_topk,
        indexer_loss=derive(cfg.indexer_loss, AscSparseIndexerLoss.Config),
    )
