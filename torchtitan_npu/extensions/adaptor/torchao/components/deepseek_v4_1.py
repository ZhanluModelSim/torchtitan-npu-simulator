# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 special-mechanism components.

Registers the V4.1-unique ``lightning_indexer``, ``compressor`` and
``sparse_attention`` plugins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import tyro
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4_1.attention import (
    CompressedSparseInnerAttention2 as DeepSeekV41CompressedSparseInnerAttention2,
)
from torchtitan_npu.models.deepseek_v4_1.attention import (
    DeepSeekV41Metadata as _DeepSeekV41Metadata,
)

from ..configs import ModelType
from ..configs.deepseek_v4_1 import V41FilterKeys
from . import QuantComponent, component

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch

    from ..recipes import RecipeContext


@component("lightning_indexer", models=(ModelType.V4_1,))
class DeepSeekV41QuantLightningIndexerComponent(QuantComponent):
    """DeepSeek-V4.1 Lightning-Indexer quantization (QLI/QSLI)."""

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        if ctx.li_quantization is None:
            return False
        if not ctx.enable_sparse_attention_quantization:
            # The quantized selector reads the entry the quantized sparse attention builds: its
            # schedules are that backend's, and the unquantized attention's entry carries a
            # different kernel's.  One without the other is a mismatch, not a degraded mode.
            raise ValueError(
                "DeepSeek-V4.1 QLI/QSLI quantization requires "
                "--extension.quantization.enable-sparse-attention-quantization: the quantized "
                "selector launches the quantized sparse attention's schedules."
            )
        if ctx.li_quantization != "mxfp4":
            raise ValueError("DeepSeek-V4.1 QLI/QSLI supports only li_quantization='mxfp4'")

        return True

    def build_converters(self, filters: dict, ctx: RecipeContext):
        from torchao_npu.configs import DeepSeekV41QuantLightningIndexerConfig

        from ..recipes import _quantization_converter

        logger.info("[Component] DeepSeekV41LightningIndexer: building QLI/QSLI converters")
        return [
            _quantization_converter(
                DeepSeekV41QuantLightningIndexerConfig(),
                filters[V41FilterKeys.LIGHTNING_INDEXER],
                model_compile_enabled=ctx.model_compile_enabled,
                unified_npu_quant_modules=ctx.unified_npu_quant_modules,
                specific_npu_quant_modules=ctx.specific_npu_quant_modules,
            )
        ]


def _sparse_attention_feature_enabled(ctx: RecipeContext, filters: dict, key: str, *, label: str) -> bool:
    """Whether the sparse-attention feature is enabled and its filter is present.

    Emits the shared "filter unavailable" warning when the configuration asks for
    sparse-attention quantization but the model has no matching filter.
    """
    if not ctx.enable_sparse_attention_quantization:
        return False
    if key not in filters:
        logger.warning(
            "Sparse attention quantization is enabled, but model type %s has no "
            "%s filter; skipping the %s replacement.",
            ctx.model_type.value,
            label,
            label,
        )
        return False
    return True


@component("compressor", models=(ModelType.V4_1,))
class DeepSeekV41QuantCompressorComponent(QuantComponent):
    """DeepSeek-V4.1 KV-source compressor quantization (fires only when present)."""

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        return _sparse_attention_feature_enabled(ctx, filters, V41FilterKeys.COMPRESSOR, label="compressor")

    def build_converters(self, filters: dict, ctx: RecipeContext):
        from torchao_npu.configs import DeepSeekV41QuantCompressorConfig

        from ..recipes import _quantization_converter

        logger.info("[Component] DeepSeekV41QuantCompressor: building compressor converter")
        return [
            _quantization_converter(
                DeepSeekV41QuantCompressorConfig(),
                filters[V41FilterKeys.COMPRESSOR],
                model_compile_enabled=ctx.model_compile_enabled,
                unified_npu_quant_modules=ctx.unified_npu_quant_modules,
                specific_npu_quant_modules=ctx.specific_npu_quant_modules,
            )
        ]


@component("sparse_attention", models=(ModelType.V4_1,))
class DeepSeekV41QuantSparseAttentionComponent(QuantComponent):
    """DeepSeek-V4.1 sparse-attention quantization (fires only when present)."""

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        return _sparse_attention_feature_enabled(ctx, filters, V41FilterKeys.SPARSE_ATTENTION, label="sparse_attention")

    def build_converters(self, filters: dict, ctx: RecipeContext):
        from torchao_npu.configs import DeepSeekV41QuantSparseAttentionConfig

        from ..recipes import _quantization_converter

        logger.info("[Component] DeepSeekV41QuantSparseAttention: building sparse-attention converter")
        # The quantized attention is its own backend: its config carries the numbers its
        # metadata is sized with and builds that metadata itself, so nothing here relies on
        # the unquantized override being installed.
        v41_geometry = _v41_attention_numbers(ctx.model_config)
        return [
            _quantization_converter(
                None,
                filters[V41FilterKeys.SPARSE_ATTENTION],
                model_compile_enabled=ctx.model_compile_enabled,
                replacement_config_type=QuantV41SparseAttention.Config,
                replacement_kwargs={
                    **v41_geometry,
                    "_torchao_npu_config": DeepSeekV41QuantSparseAttentionConfig(),
                },
                unified_npu_quant_modules=ctx.unified_npu_quant_modules,
                specific_npu_quant_modules=ctx.specific_npu_quant_modules,
            )
        ]


@dataclass(frozen=True, kw_only=True, slots=True)
class _V41QuantAttentionMetadata(_DeepSeekV41Metadata):
    """The row, this geometry's frames, and the quantized kernels' metadata.

    Same interface as the unquantized stack's entry (``aicpu_metadata.sparse_attn`` /
    ``.lightning_indexer``), built by this package because the quantized attention is its own
    backend: it is the fused attention's copy with the sliding-window KV fake-quantized.  The
    row and its frames are the model metadata's; this entry adds only the operator outputs.
    """

    aicpu_metadata: Any

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        attention_metadata: _V41QuantAttentionMetadata,
        *,
        permutation,
    ):
        """The quantized stack does not rebuild its metadata per rank yet (context parallelism)."""
        del attention_metadata, permutation
        raise ValueError(
            "the quantized sparse attention does not rebuild its metadata per rank yet: run it "
            "with --parallelism.context-parallel-degree 1."
        )


class QuantV41SparseAttention(DeepSeekV41CompressedSparseInnerAttention2):
    """The quantized sparse attention: the reference attention's node, its forward swapped.

    This is the *node* a recipe's config ends up with; the forward it runs lives in
    ``torchao_npu.quantized_modules.deepseek_v4_1.sparse_attention``, which happens to give its
    class the same name and is swapped in by ``quantize_`` below.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41CompressedSparseInnerAttention2.Config):
        """The reference attention plus the numbers this backend's metadata is sized with."""

        num_heads: int
        head_dim: int
        index_topk: int
        index_num_heads: int
        index_head_dim: int
        candidate_topk_blocks: int
        candidate_block_size: int
        _torchao_npu_config: Annotated[Any, tyro.conf.Suppress] = None

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> _V41QuantAttentionMetadata:
            """The row this backend already builds, plus its own kernels' metadata."""
            from torchao_npu.quantized_modules.deepseek_v4_1.attention_metadata import (
                build_attention_metadata as build_quant_metadata,
            )

            # ``slots=True`` recreates the dataclass, so the parent is named explicitly rather
            # than reached through a zero-argument ``super()``.
            row = DeepSeekV41CompressedSparseInnerAttention2.Config.build_attention_metadata(
                self,
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            q, swa_k, cmp_k = row.q, row.swa_k, row.cmp_k
            return _V41QuantAttentionMetadata(
                cu_seq_q=row.cu_seq_q,
                cu_seq_k=row.cu_seq_k,
                position_k=row.position_k,
                restore=row.restore,
                ratio=self.compress_ratio,
                q=q,
                swa_k=swa_k,
                cmp_k=cmp_k,
                aicpu_metadata=build_quant_metadata(
                    ratio=self.compress_ratio,
                    q=q,
                    swa_k=swa_k,
                    cmp_k=cmp_k,
                    window_size=self.window_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    index_topk=self.index_topk,
                    index_num_heads=self.index_num_heads,
                    index_head_dim=self.index_head_dim,
                    topk_blocks=self.candidate_topk_blocks,
                    block_size=self.candidate_block_size,
                    num_rows=positions.numel(),
                ),
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if config._torchao_npu_config is None:
            raise ValueError(f"{type(self).__name__}.Config requires _torchao_npu_config")
        from torchao.quantization import quantize_

        quantize_(
            self,
            config._torchao_npu_config,
            filter_fn=lambda candidate, _fqn: candidate is self,
        )


def _v41_attention_numbers(model_config) -> dict[str, int]:
    """The numbers the quantized attention's metadata is sized with, off the model config.

    One config describes one geometry: ``test_fused_geometry`` pins the same five widths (plus the
    candidate pool) on every layer, so they are read once here rather than restated on the
    replacement node by each layer.  Mixed stacks and disagreeing pools are refused instead.
    """
    layers = list(model_config.layers)
    attention = layers[0].attention
    indexer = attention.indexer
    first = (
        attention.n_heads,
        attention.head_dim,
        indexer.index_topk,
        indexer.num_index_heads,
        indexer.index_head_dim,
    )
    for layer in layers[1:]:
        other = (
            layer.attention.n_heads,
            layer.attention.head_dim,
            layer.attention.indexer.index_topk,
            layer.attention.indexer.num_index_heads,
            layer.attention.indexer.index_head_dim,
        )
        if other != first:
            raise ValueError(
                f"the stack mixes geometries ({first} vs {other}), which one quantized attention config cannot describe"
            )
    # The pool is declared per layer, on the selector that builds or searches it, so it is read
    # there; the model-level pair is the same statement and is checked against it rather than
    # silently preferred.  A missing field is an error, not a default: ``-1`` here would size the
    # metadata for no pool while the launches still pass one.
    pools = {
        (layer.attention.indexer.selector.candidate_topk_blocks, layer.attention.indexer.selector.candidate_block_size)
        for layer in layers
    }
    positive = {(blocks, size) for blocks, size in pools if blocks > 0}
    if len(positive) > 1:
        raise ValueError(f"the stack declares more than one candidate pool: {sorted(positive)}")
    pool = positive.pop() if positive else (-1, -1)
    declared = (int(model_config.candidate_topk_blocks), int(model_config.candidate_block_size))
    if pool[0] > 0 and declared != pool:
        raise ValueError(
            f"the candidate pool the layers declare ({pool}) disagrees with the model config's "
            f"({declared}): state one value."
        )
    return {
        "num_heads": first[0],
        "head_dim": first[1],
        "index_topk": first[2],
        "index_num_heads": first[3],
        "index_head_dim": first[4],
        "candidate_topk_blocks": pool[0],
        "candidate_block_size": pool[1],
    }


if _V41QuantAttentionMetadata not in SUPPORTED_NODES:
    register_dataclass(
        _V41QuantAttentionMetadata,
        serialized_type_name=f"{_V41QuantAttentionMetadata.__module__}.{_V41QuantAttentionMetadata.__name__}",
    )
