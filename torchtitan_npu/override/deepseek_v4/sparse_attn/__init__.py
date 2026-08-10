"""DeepSeek-V4 sparse-attention overrides (registry-facing module).

The CANN fused implementation (metadata layer + kernel) lives in
``cann.py`` and the eager golden reference in ``golden.py``; the
registrations are defined here so the override paths stay
``override.deepseek_v4.sparse_attn.{cann_metadata, cann, golden}``.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import CompressedBlockMaskHandler

from .cann import (
    CANNCompressedSparseInnerAttention,
    CANNCompressedVarlenMetadataHandler,
)
from .golden import GoldenCompressedSparseInnerAttention


@override(
    target=CompressedBlockMaskHandler.Config,
    description=(
        "DSV4 varlen contract plus the precomputed CANN sparse-attention "
        "metadata kernels"
    ),
)
def cann_metadata(
    cfg: CompressedBlockMaskHandler.Config,
    *,
    num_heads: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
) -> CANNCompressedVarlenMetadataHandler.Config:
    return derive(
        cfg,
        CANNCompressedVarlenMetadataHandler.Config,
        num_heads=num_heads,
        head_dim=head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
    )


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description=(
        "Use cann_ops_transformer TND LightningIndexer and SparseFlashMLA "
        "with fused SASG/SLIG backward"
    ),
)
def cann(
    cfg: CompressedSparseInnerAttention.Config,
    indexer_loss_coeff: float = 1.0,
) -> CANNCompressedSparseInnerAttention.Config:
    # ``sharding_config`` is copied from the source by ``derive``; the
    # sharding pass already ran before the overrides (update_from_config),
    # so the derived config keeps the sharding-pass values as-is.
    return derive(
        cfg,
        CANNCompressedSparseInnerAttention.Config,
        indexer_loss_coeff=indexer_loss_coeff,
    )


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description=(
        "DSV4 sparse attention golden reference "
        "(eager per-document, dsv4-infer-npu bitwise)"
    ),
)
def golden(
    cfg: CompressedSparseInnerAttention.Config,
) -> GoldenCompressedSparseInnerAttention.Config:
    return derive(cfg, GoldenCompressedSparseInnerAttention.Config)
