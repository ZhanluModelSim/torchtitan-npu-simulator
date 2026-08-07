# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 DSA with fused CANN TND kernels.

This replaces the attention kernel with LightningIndexer and SparseFlashMLA.
It requires the packed metadata handler from ``varlen_dsa``.
"""

import dataclasses as dc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.config import override

from torchtitan_npu.models.deepseek_v4.attention import DSAFlexAttention
from torchtitan_npu.models.deepseek_v4.packed import (
    CompressionInfo,
    DSV4PackedMetadata,
    compact_compressed_tensor,
    compact_token_tensor,
    restore_token_tensor,
)
from torchtitan_npu.ops.cann_transformer import get_cann_transformer_ops
from torchtitan_npu.override.deepseek_v4.varlen_dsa import (
    DSAVarlenAttention,
    derive_varlen_dsa,
)

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


class SMLAMetadataCache:
    """Cache opaque CANN metadata for one attention module and microbatch."""

    def __init__(self) -> None:
        self._batch_cache_id: int | None = None
        # Cache key presence separately because valid metadata may be None.
        self._values: dict[tuple[str, int, tuple[Any, ...]], torch.Tensor | None] = {}

    def get_or_create(
        self,
        metadata: DSV4PackedMetadata,
        name: str,
        ratio: int,
        signature: tuple[Any, ...],
        builder: Callable[[], torch.Tensor | None],
    ) -> torch.Tensor | None:
        if torch.compiler.is_compiling():
            # AOTAutograd cannot trace this cache mutation from custom backward.
            return builder()
        if self._batch_cache_id != metadata.cache_id:
            self._batch_cache_id = metadata.cache_id
            self._values.clear()
        key = (name, ratio, signature)
        if key not in self._values:
            self._values[key] = builder()
        return self._values[key]


def _compressed_metadata(
    metadata: DSV4PackedMetadata,
    ratio: int,
) -> CompressionInfo | None:
    return None if ratio <= 1 else metadata.compression_for_ratio(ratio)


def _validate_model_input_layout(
    query: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor | None,
    metadata: DSV4PackedMetadata | None,
    ratio: int,
) -> tuple[DSV4PackedMetadata, CompressionInfo | None]:
    """Validate model-facing BSND/BSD inputs before compacting them to TND."""

    if not isinstance(metadata, DSV4PackedMetadata):
        raise TypeError("npu_smla_tnd requires DSV4PackedMetadata attention masks.")
    if query.ndim != 4 or ori_kv.ndim != 3:
        raise ValueError(
            "npu_smla_tnd expects query=[B,S,N,D] and original KV=[B,S,D]."
        )
    batch_size, seqlen = query.shape[:2]
    if ori_kv.shape[:2] != (batch_size, seqlen):
        raise ValueError("npu_smla_tnd query and original-KV shapes disagree.")
    if (
        metadata.container_batch_size != batch_size
        or metadata.container_seq_len != seqlen
    ):
        raise ValueError("npu_smla_tnd metadata does not describe the model tensors.")

    compressed = _compressed_metadata(metadata, ratio)
    if ratio <= 1:
        if cmp_kv is not None and cmp_kv.numel() != 0:
            raise ValueError("ratio-1 npu_smla_tnd must not receive compressed KV.")
        return metadata, None

    if cmp_kv is None or cmp_kv.ndim != 3:
        raise ValueError("ratio>1 npu_smla_tnd requires compressed KV=[B,C,D].")
    expected = (batch_size, seqlen // ratio)
    if cmp_kv.shape[:2] != expected:
        raise ValueError("Compressed KV storage must be [B,S//ratio,D] for local TND.")
    return metadata, compressed


def _common_metadata_kwargs(
    metadata: DSV4PackedMetadata,
    compressed: CompressionInfo | None,
    *,
    ratio: int,
    topk: int,
    window_size: int,
) -> dict[str, Any]:
    return {
        # Original KV uses query boundaries; compressed KV uses block boundaries.
        "cu_seqlens_q": metadata.varlen.cu_seq_q,
        "cu_seqlens_ori_kv": metadata.varlen.cu_seq_k,
        "cu_seqlens_cmp_kv": (
            None if compressed is None else compressed.varlen.cu_seq_k
        ),
        "cmp_residual_kv": None if compressed is None else compressed.residual,
        "ori_topk_length": None,
        "cmp_topk_length": None,
        "ori_topk": 0,
        "cmp_topk": topk if ratio == 4 else 0,
        "cmp_ratio": ratio,
        "ori_mask_mode": _ORI_MASK_MODE,
        "cmp_mask_mode": _CMP_MASK_MODE,
        "ori_win_left": window_size - 1,
        "ori_win_right": 0,
        "layout_q": _LAYOUT,
        "layout_kv": _LAYOUT,
        "has_ori_kv": True,
        "has_cmp_kv": compressed is not None,
    }


def _sparse_metadata(
    ops: Any,
    cache: SMLAMetadataCache,
    metadata: DSV4PackedMetadata,
    compressed: CompressionInfo | None,
    *,
    num_heads_q: int,
    head_dim: int,
    ratio: int,
    topk: int,
    window_size: int,
) -> torch.Tensor | None:
    signature = (
        str(metadata.varlen.cu_seq_q.device),
        num_heads_q,
        head_dim,
        topk,
        window_size,
    )
    kwargs = _common_metadata_kwargs(
        metadata,
        compressed,
        ratio=ratio,
        topk=topk,
        window_size=window_size,
    )
    return cache.get_or_create(
        metadata,
        "sparse_flash_mla",
        ratio,
        signature,
        lambda: ops.sparse_flash_mla_metadata(num_heads_q, 1, head_dim, **kwargs),
    )


def _lightning_indexer_metadata(
    ops: Any,
    cache: SMLAMetadataCache,
    metadata: DSV4PackedMetadata,
    compressed: CompressionInfo,
    *,
    num_heads: int,
    head_dim: int,
    topk: int,
) -> torch.Tensor | None:
    signature = (
        str(metadata.varlen.cu_seq_q.device),
        num_heads,
        head_dim,
        topk,
    )
    return cache.get_or_create(
        metadata,
        "lightning_indexer",
        4,
        signature,
        lambda: ops.lightning_indexer_metadata(
            num_heads,
            1,
            head_dim,
            topk,
            cu_seqlens_q=metadata.varlen.cu_seq_q,
            cu_seqlens_k=compressed.varlen.cu_seq_k,
            cmp_residual_k=compressed.residual,
            layout_q=_LAYOUT,
            layout_k=_LAYOUT,
            mask_mode=_CMP_MASK_MODE,
            cmp_ratio=4,
        ),
    )


def _normalize_li_indices(
    indices: torch.Tensor,
    *,
    total_query_tokens: int,
    topk: int,
) -> torch.Tensor:
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    if indices.shape != (total_query_tokens, 1, topk):
        raise RuntimeError(
            "cann_ops_transformer LightningIndexer returned an unexpected shape: "
            f"expected {(total_query_tokens, 1, topk)}, got {tuple(indices.shape)}."
        )
    return indices.to(torch.int32).contiguous()


def _run_lightning_indexer(
    ops: Any,
    cache: SMLAMetadataCache,
    metadata: DSV4PackedMetadata,
    compressed: CompressionInfo,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    op_metadata = _lightning_indexer_metadata(
        ops,
        cache,
        metadata,
        compressed,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        topk=topk,
    )
    indices, _ = ops.lightning_indexer(
        q.to(torch.bfloat16).contiguous(),
        k.to(torch.bfloat16).contiguous(),
        weights.float().contiguous(),
        topk,
        cu_seqlens_q=metadata.varlen.cu_seq_q,
        cu_seqlens_k=compressed.varlen.cu_seq_k,
        cmp_residual_k=compressed.residual,
        metadata=op_metadata,
        layout_q=_LAYOUT,
        layout_k=_LAYOUT,
        mask_mode=_CMP_MASK_MODE,
        cmp_ratio=4,
        return_value=1,
    )
    return _normalize_li_indices(
        indices,
        total_query_tokens=q.shape[0],
        topk=topk,
    )


def _compute_li_loss(
    indexer_softmax: torch.Tensor,
    teacher_mass: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Recover the reference LI loss from SLIG outputs."""
    student = indexer_softmax.float().clamp_min(1e-10)
    target = teacher_mass.float().clamp_min(0)
    target_sum = target.sum(dim=-1, keepdim=True)
    valid_target = target_sum > 1e-10
    student = torch.where(valid_target, student, torch.ones_like(student))
    teacher = target / target_sum.clamp_min(1e-10)
    log_teacher = teacher.clamp_min(1e-10).log()
    loss = (teacher * (log_teacher - student.log())).sum(dim=-1)
    return (target_sum.squeeze(-1) * loss).mean() * softmax_scale


class _SparseFlashMLATND(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        indexer_q,
        indexer_k,
        index_weights,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        softmax_scale,
        ratio,
        window_size,
        topk,
        indexer_loss_coeff,
        op_metadata,
        indexer_loss_accumulator,
    ):
        ops = get_cann_transformer_ops()
        has_compressed = ratio > 1
        has_sparse_indices = topk > 0
        result, softmax_lse = ops.sparse_flash_mla(
            query,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv if has_compressed else None,
            cmp_sparse_indices=(cmp_sparse_indices if has_sparse_indices else None),
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            sinks=sinks,
            metadata=op_metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=_CMP_MASK_MODE,
            ori_win_left=window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            result,
            softmax_lse,
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            sinks,
            indexer_q,
            indexer_k,
            index_weights,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
        )
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        ctx.window_size = window_size
        ctx.topk = topk
        ctx.indexer_loss_coeff = indexer_loss_coeff
        ctx.indexer_loss_accumulator = indexer_loss_accumulator
        return result

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        (
            result,
            softmax_lse,
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            sinks,
            indexer_q,
            indexer_k,
            index_weights,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
        ) = ctx.saved_tensors
        ops = get_cann_transformer_ops()
        has_compressed = ctx.ratio > 1
        has_sparse_indices = ctx.topk > 0
        grad_metadata = ops.sparse_flash_mla_grad_metadata(
            query.shape[1],
            1,
            query.shape[2],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            ori_topk=0,
            cmp_topk=ctx.topk if ctx.ratio == 4 else 0,
            cmp_ratio=ctx.ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=_CMP_MASK_MODE,
            ori_win_left=ctx.window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
            has_ori_kv=True,
            has_cmp_kv=has_compressed,
        )
        (
            dquery,
            dori_kv,
            dcmp_kv,
            dsinks,
            _,
            cmp_softmax_l1,
        ) = ops.sparse_flash_mla_grad(
            query,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv if has_compressed else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=(cmp_sparse_indices if has_sparse_indices else None),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=grad_metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=_CMP_MASK_MODE,
            ori_win_left=ctx.window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
        )
        if not has_compressed:
            dcmp_kv = None

        dindexer_q = dindexer_k = dindex_weights = None
        if ctx.ratio == 4 and ctx.indexer_loss_coeff != 0:
            if any(x is None for x in (indexer_q, indexer_k, index_weights)):
                raise RuntimeError(
                    "ratio-4 npu_smla_tnd requires LI tensors in backward."
                )
            slig_metadata = ops.sparse_lightning_indexer_kl_loss_grad_metadata(
                indexer_q.shape[1],
                1,
                indexer_q.shape[2],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_cmp_kv,
                cmp_residual_k=cmp_residual_kv,
                topk=ctx.topk,
                layout_q=_LAYOUT,
                layout_k=_LAYOUT,
                mask_mode=_CMP_MASK_MODE,
                cmp_ratio=4,
            )
            dindexer_q, dindexer_k, dindex_weights, indexer_softmax = (
                ops.sparse_lightning_indexer_kl_loss_grad(
                    q=indexer_q,
                    k=indexer_k,
                    w=index_weights.float(),
                    sparse_indices=cmp_sparse_indices,
                    attn_softmax_l1_norm=cmp_softmax_l1,
                    cmp_residual_k=cmp_residual_kv,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_cmp_kv,
                    metadata=slig_metadata,
                    layout_q=_LAYOUT,
                    layout_k=_LAYOUT,
                    mask_mode=_CMP_MASK_MODE,
                    cmp_ratio=4,
                )
            )
            if ctx.indexer_loss_accumulator is not None:
                li_loss = _compute_li_loss(
                    indexer_softmax,
                    cmp_softmax_l1,
                    ctx.softmax_scale,
                )
                ctx.indexer_loss_accumulator.add_(li_loss.detach())
            query_rows = cmp_softmax_l1.sum(dim=-1).numel()
            grad_scale = ctx.indexer_loss_coeff * ctx.softmax_scale / float(query_rows)
            dindexer_q = (dindexer_q * grad_scale).to(indexer_q.dtype)
            dindexer_k = (dindexer_k * grad_scale).to(indexer_k.dtype)
            dindex_weights = (dindex_weights * grad_scale).to(index_weights.dtype)

        return (
            dquery,
            dori_kv,
            dcmp_kv,
            None,
            dsinks,
            dindexer_q,
            dindexer_k,
            dindex_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class NPUSMLATNDAttention(DSAVarlenAttention):
    """Run LI and SMLA/SMLAG/SLIG in a local TND layout."""

    @dataclass(kw_only=True, slots=True)
    class Config(DSAVarlenAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        get_cann_transformer_ops()
        self._smla_metadata_cache = SMLAMetadataCache()
        self.index_topk = (
            0 if config.index_selection is None else config.index_selection.index_topk
        )

    def forward(
        self,
        query_states,
        kv_states,
        kv_compress,
        attn_sink,
        q_indexer=None,
        k_indexer=None,
        index_weights=None,
        *,
        attention_masks=None,
    ):
        metadata, compressed = _validate_model_input_layout(
            query_states,
            kv_states,
            kv_compress,
            attention_masks,
            self.compress_ratio,
        )
        query = compact_token_tensor(query_states, metadata).contiguous()
        ori_kv = compact_token_tensor(kv_states, metadata).unsqueeze(1).contiguous()
        cmp_kv = (
            None
            if compressed is None
            else compact_compressed_tensor(kv_compress, metadata, self.compress_ratio)
            .unsqueeze(1)
            .contiguous()
        )

        index_q = index_k = weights = cmp_sparse_indices = None
        if self.compress_ratio == 4:
            if q_indexer is None or k_indexer is None or index_weights is None:
                raise ValueError(
                    "ratio-4 npu_smla_tnd requires all LI projection tensors."
                )
            assert compressed is not None
            index_q = compact_token_tensor(q_indexer, metadata).contiguous()
            index_k = (
                compact_compressed_tensor(k_indexer, metadata, self.compress_ratio)
                .unsqueeze(1)
                .contiguous()
            )
            weights = compact_token_tensor(index_weights, metadata).contiguous()
            cmp_sparse_indices = _run_lightning_indexer(
                get_cann_transformer_ops(),
                self._smla_metadata_cache,
                metadata,
                compressed,
                index_q,
                index_k,
                weights,
                self.index_topk,
            )

        indexer_loss_coeff = 0.0
        indexer_loss_accumulator = None
        if self.training and hasattr(self, "indexer_aux_loss"):
            indexer_loss_coeff = float(self.indexer_aux_loss.coeff)
            # Backward records this once per microbatch, including with AC.
            indexer_loss_accumulator = self.indexer_aux_loss._acc

        ops = get_cann_transformer_ops()
        op_metadata = _sparse_metadata(
            ops,
            self._smla_metadata_cache,
            metadata,
            compressed,
            num_heads_q=query.shape[1],
            head_dim=query.shape[2],
            ratio=self.compress_ratio,
            topk=self.index_topk,
            window_size=self.window_size,
        )
        empty_value = query.new_empty((0,))
        empty_int32 = query.new_empty((0,), dtype=torch.int32)
        output = _SparseFlashMLATND.apply(
            query,
            ori_kv,
            cmp_kv if cmp_kv is not None else empty_value.clone(),
            (
                cmp_sparse_indices
                if cmp_sparse_indices is not None
                else empty_int32.clone()
            ),
            attn_sink.float().contiguous(),
            index_q if index_q is not None else empty_value.clone(),
            index_k if index_k is not None else empty_value.clone(),
            weights if weights is not None else empty_value.clone(),
            metadata.varlen.cu_seq_q,
            metadata.varlen.cu_seq_q.clone(),
            (empty_int32.clone() if compressed is None else compressed.varlen.cu_seq_k),
            empty_int32.clone() if compressed is None else compressed.residual,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            self.index_topk,
            indexer_loss_coeff,
            op_metadata,
            indexer_loss_accumulator,
        )
        return restore_token_tensor(output, metadata).contiguous()


def _tensor_output_sharding(cfg: DSAFlexAttention.Config):
    sharding = cfg.sharding_config
    if sharding is None:
        return None
    out_src = sharding.out_src_shardings
    if isinstance(out_src, tuple):
        out_src = out_src[0]
    out_dst = sharding.out_dst_shardings
    if out_dst is None and cfg.return_lse:
        out_dst = out_src
    return dc.replace(
        sharding,
        out_src_shardings=out_src,
        out_dst_shardings=out_dst,
    )


@override(
    target=DSAFlexAttention.Config,
    exact=True,
    description=(
        "Use cann_ops_transformer TND LightningIndexer and SparseFlashMLA "
        "with fused SASG/SLIG backward"
    ),
)
def npu_smla_tnd_override(
    cfg: DSAFlexAttention.Config,
) -> NPUSMLATNDAttention.Config:
    return derive_varlen_dsa(
        cfg,
        NPUSMLATNDAttention.Config,
        return_lse=False,
        sharding_config=_tensor_output_sharding(cfg),
    )
