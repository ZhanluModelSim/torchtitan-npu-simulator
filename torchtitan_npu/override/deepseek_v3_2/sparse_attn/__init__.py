"""DeepSeek-V3.2 sparse-attention overrides (registry-facing module).

The AscendC implementation (fused attention + the TND metadata its config
builds) lives in ``ascendc.py``; the registration is defined here so the
override path stays ``override.deepseek_v3_2.sparse_attn.asc``.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v3_2.model import SparseInnerAttention

from .ascendc import (
    AscSparseIndexerLoss,
    AscSparseInnerAttention,
)


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
