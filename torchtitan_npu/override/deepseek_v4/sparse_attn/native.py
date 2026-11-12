# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SMLA inner-attention adapters for TorchTitan's unmodified V4 attention."""

from dataclasses import dataclass, fields
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    create_varlen_metadata_for_document,
)
from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    DSV4InnerAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer

from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata
from torchtitan_npu.override.deepseek_v4.metadata import AicpuGeometry, AscV4Metadata


@dataclass(kw_only=True, slots=True)
class NativeAscV4Metadata(AscV4Metadata):
    """Native document offsets alongside the existing CANN kernel records."""

    native: VarlenAttentionMetadata

    @property
    def cu_seq_q(self):
        return self.native.cu_seq_q

    @property
    def cu_seq_k(self):
        return self.native.cu_seq_k

    @property
    def max_q(self):
        return self.native.max_q

    @property
    def max_k(self):
        return self.native.max_k

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        del attention_metadata, permutation
        raise NotImplementedError("Native V4 SMLA adapters currently require CP=1.")


if NativeAscV4Metadata not in SUPPORTED_NODES:
    register_dataclass(
        NativeAscV4Metadata,
        serialized_type_name=f"{NativeAscV4Metadata.__module__}.{NativeAscV4Metadata.__name__}",
    )


@dataclass(kw_only=True, slots=True)
class AscNativeInnerAttentionConfig(DSV4InnerAttention.Config):
    num_heads: int
    head_dim: int
    index_num_heads: int
    index_head_dim: int
    selector_backend: Literal["native", "asc"] = "native"

    # The upstream annotation names its concrete NamedTuple. This backend's
    # registered dataclass carries those same native fields plus CANN records.
    def build_attention_metadata(  # pyrefly: ignore [bad-override]
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> NativeAscV4Metadata:
        # Importing/configuring the backend does not import the CANN extension.
        from .ascendc import build_aicpu_metadata

        native = create_varlen_metadata_for_document(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )
        # Native fixed-capacity offsets may include empty reserved segments.
        # CANN describes only nonempty documents; removing repeated boundaries
        # changes neither token order nor any nonempty document's boundaries.
        with spmd.no_typecheck():
            positive = native.cu_seq_q[1:] > native.cu_seq_q[:-1]
            kernel_cu = torch.cat((native.cu_seq_q[:1], native.cu_seq_q[1:][positive]))
        # The existing CANN backend requires each packed segment to be aligned
        # to its compression ratio; the builder validates this explicitly.
        row = build_v4_metadata(positions, max(self.compress_ratio, 1), cu_seq_q=kernel_cu, build_slots=False)
        geometry = AicpuGeometry(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            window_size=self.window_size,
            index_topk=self.index_topk,
            index_num_heads=self.index_num_heads,
            index_head_dim=self.index_head_dim,
        )
        fused = build_aicpu_metadata(
            row,
            geometry=geometry,
            positions_q=positions,
            include_selector=self.selector_backend == "asc" and self.compress_ratio == 4,
        )
        return NativeAscV4Metadata(
            **{item.name: getattr(fused, item.name) for item in fields(AscV4Metadata)}, native=native
        )


class _AscNativeAttention(DSV4InnerAttention):
    def _parallelize(self, parallelism_context):
        if parallelism_context.get_optional_mesh("tp") is not None:
            raise NotImplementedError("Native V4 SMLA adapters currently require TP=1.")
        if parallelism_context.get_optional_mesh("cp") is not None:
            raise NotImplementedError("Native V4 SMLA adapters currently require CP=1.")
        super()._parallelize(parallelism_context)

    def _run_kernel(self, q, swa_k, attn_sink, attention_metadata, *, cmp_k=None, indices=None, carrier=None):
        from .ascendc import run_sparse_flash_mla

        if not isinstance(attention_metadata, NativeAscV4Metadata):
            raise TypeError("Native V4 SMLA requires the NativeAscV4Metadata its config built.")
        if attention_metadata.cp is not None:
            raise NotImplementedError("Native V4 SMLA adapters currently require CP=1.")
        if attention_metadata.ratio != max(self.compress_ratio, 1):
            raise ValueError("The attention metadata describes a different compression ratio.")
        with spmd.no_typecheck():
            output, lse = run_sparse_flash_mla(
                q,
                swa_k,
                attn_sink,
                attention_metadata=attention_metadata,
                softmax_scale=self.softmax_scale,
                compress_ratio=self.compress_ratio,
                window_size=self.window_size,
                cmp_k=cmp_k,
                topk_indices=indices,
                teacher_carrier=carrier,
            )
        if spmd.is_type_checking():
            spmd.assert_type(output, spmd.get_local_type(q), spmd.get_partition_spec(q))
        return output, lse


class AscSlidingWindowAttention(_AscNativeAttention, SlidingWindowAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(AscNativeInnerAttentionConfig, SlidingWindowAttention.Config):
        pass

    def forward(self, q, swa_k, attn_sink, *, attention_metadata=None):
        output, _lse = self._run_kernel(q, swa_k, attn_sink, attention_metadata)
        return output


class AscHeavilyCompressedAttention(_AscNativeAttention, HeavilyCompressedAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(AscNativeInnerAttentionConfig, HeavilyCompressedAttention.Config):
        pass

    def forward(self, q, swa_k, cmp_k, attn_sink, *, attention_metadata=None):
        # SMLA implements HCA's full causal compressed prefix directly.
        output, _lse = self._run_kernel(q, swa_k, attn_sink, attention_metadata, cmp_k=cmp_k)
        return output


class AscCompressedSparseAttention(_AscNativeAttention, CompressedSparseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(AscNativeInnerAttentionConfig, CompressedSparseAttention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.selector_backend = config.selector_backend
        self.selector = None
        if config.selector_backend == "asc":
            if config.aux_loss is not None:
                raise ValueError("selector_backend='asc' has its own indexer objective; set aux_loss=None.")
            from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import AscLightningIndexer

            self.selector = AscLightningIndexer.Config(index_topk=config.index_topk).build()
        elif config.selector_backend != "native":
            raise ValueError(f"Unknown V4 selector backend: {config.selector_backend!r}.")

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        idx_q,
        idx_k,
        idx_w,
        attn_sink,
        *,
        attention_metadata=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ):
        if not isinstance(attention_metadata, NativeAscV4Metadata):
            raise TypeError("Native V4 CSA requires the NativeAscV4Metadata its config built.")
        with spmd.no_typecheck():
            if self.selector is None:
                # Native CSA calls this class-static function, rather than the
                # projection Indexer instance. Keep that choice and its logits.
                indices, scores = Indexer.select(
                    idx_q,
                    idx_k,
                    idx_w,
                    max_seqlen=attention_metadata.max_k,
                    ratio=self.compress_ratio,
                    topk=self.index_topk,
                    cu_seqlens=attention_metadata.cu_seq_q,
                    return_scores=self.training and self.aux_loss is not None,
                )
            else:
                indices, scores = self.selector(idx_q, idx_k, idx_w, attention_metadata=attention_metadata)
        # Native selection bounds K by the available pool. CANN records are
        # sized with the configured K, so append invalid slots for the kernel
        # while retaining the original indices/logits for the native loss.
        kernel_indices = indices
        if self.selector is None and indices.shape[-1] < self.index_topk:
            kernel_indices = F.pad(indices, (0, self.index_topk - indices.shape[-1]), value=-1)
        output, lse = self._run_kernel(
            q,
            swa_k,
            attn_sink,
            attention_metadata,
            cmp_k=cmp_k,
            indices=kernel_indices,
            carrier=scores if self.training and self.selector is not None else None,
        )
        if self.training and self.aux_loss is not None:
            if aux_loss_denominator is None:
                raise RuntimeError("SparseIndexerLoss requires aux_loss_denominator.")
            assert scores is not None
            # CANN TND LSE is [KV heads, T, Q heads / KV heads], here [1,T,H].
            # TorchTitan's teacher consumes [T,H], without a transpose.
            with spmd.no_typecheck():
                output = self.aux_loss(
                    q.detach(),
                    cmp_k.detach(),
                    indices,
                    scores,
                    lse.squeeze(0).detach(),
                    attention_metadata.cu_seq_q,
                    carrier=output,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask_T,
                )
            if spmd.is_type_checking():
                spmd.assert_type(output, spmd.get_local_type(q), spmd.get_partition_spec(q))
        return output
