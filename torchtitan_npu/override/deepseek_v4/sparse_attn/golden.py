# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: the DeepSeek-V4 numerical-reference DSA (eager, per-document).

An eager per-document implementation over the packed stream, matching the
``dsv4-infer-npu`` inference baseline op-for-op (FP32 gather-matmul
attention with a per-head sink, per-document indexer top-k).  It anchors the
two-layer numeric scheme: this golden is compared bitwise against patched
transformers, and the fused CANN kernels are checked within tolerance of it.
"""

import itertools
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import CompressedVarlenMetadata
from torchtitan_npu.models.deepseek_v4.reference import (
    ReferenceCompressedVarlenMetadata,
)

# Bound the ``[B, M, topk, D]`` gather used by sparse attention.
_ATTN_CHUNK = int(os.environ.get("TTNPU_DSA_ATTN_CHUNK", "256"))


def _window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    device,
) -> torch.Tensor:
    window = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (base - window + 1).clamp(0) + torch.arange(window, device=device)
    idxs = torch.where(idxs > base, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def _sparse_attn_chunk(
    q_BMHD: torch.Tensor,
    kv_BND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    topk_idxs_BMK: torch.Tensor,
    softmax_scale: float,
    *,
    return_lse: bool,
):
    """Match the baseline sparse-attention chunk with a per-head sink."""
    b, m, k = topk_idxs_BMK.shape

    # Map unused ``-1`` slots to row 0, then mask their values.
    valid_BMK = topk_idxs_BMK != -1
    safe_BMK = torch.where(valid_BMK, topk_idxs_BMK, torch.zeros_like(topk_idxs_BMK)).long()
    batch_BMK = torch.arange(b, device=kv_BND.device).view(b, 1, 1).expand(-1, m, k)
    kv_BMKD = kv_BND[batch_BMK, safe_BMK, :] * valid_BMK.unsqueeze(-1).to(kv_BND.dtype)

    scores_BMHK = torch.matmul(q_BMHD.float(), kv_BMKD.float().transpose(-2, -1))
    scores_BMHK = scores_BMHK * softmax_scale
    scores_BMHK = scores_BMHK.masked_fill(~valid_BMK.unsqueeze(-2), -torch.inf)

    # Include the learned per-head sink in the softmax denominator.
    max_BMH1 = scores_BMHK.max(dim=-1, keepdim=True).values
    exp_BMHK = torch.exp(scores_BMHK - max_BMH1)
    sum_BMH1 = exp_BMHK.sum(dim=-1, keepdim=True)
    sink_BMH1 = torch.exp(attn_sink_H.view(1, 1, -1, 1).float() - max_BMH1)
    probs_BMHK = exp_BMHK / (sum_BMH1 + sink_BMH1)

    out_BMHD = torch.matmul(probs_BMHK, kv_BMKD.to(torch.float32))
    if not return_lse:
        return out_BMHD

    # Include the sink in the LSE consumed by the indexer auxiliary loss.
    lse_BMH = (max_BMH1 + torch.log(sum_BMH1 + sink_BMH1)).squeeze(-1)
    return out_BMHD, lse_BMH


def _sparse_attn(
    q_BMHD: torch.Tensor,
    kv_BND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    topk_idxs_BMK: torch.Tensor,
    softmax_scale: float,
    *,
    return_lse: bool,
):
    m = q_BMHD.size(1)
    chunk = _ATTN_CHUNK if _ATTN_CHUNK > 0 else m
    outs, lses = [], []
    for i in range(0, m, chunk):
        res = _sparse_attn_chunk(
            q_BMHD[:, i : i + chunk],
            kv_BND,
            attn_sink_H,
            topk_idxs_BMK[:, i : i + chunk],
            softmax_scale,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = res
            outs.append(out)
            lses.append(lse)
        else:
            outs.append(res)
    out_BMHD = torch.cat(outs, dim=1)
    if return_lse:
        return out_BMHD, torch.cat(lses, dim=1)
    return out_BMHD


def _sequence_ranges(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    """Materialize host-side sequence ranges for the eager document loops.

    Only this reference path needs Python bounds; the fused kernels take
    ``cu_seqlens`` directly.
    """

    bounds = cu_seqlens.tolist()
    return list(itertools.pairwise(bounds))


def _packed_block_docs(plan, total_blocks: int, device) -> torch.Tensor:
    """Document id of every packed block (0-based, int64)."""
    if total_blocks == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    return torch.searchsorted(
        plan.cu_seqlens_cmp_k[1:],
        torch.arange(total_blocks, device=device),
        right=True,
    )


class GoldenCompressedSparseInnerAttention(CompressedSparseInnerAttention):
    """Reference DSA sparse attention over packed varlen metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention.Config):
        pass

    def _packed_compressed_indices(
        self,
        metadata: CompressedVarlenMetadata,
        device,
    ) -> torch.Tensor:
        """HCA reference: per-document causal compressed-block indices."""
        plan = metadata.plans[self.compress_ratio]
        block_ranges = _sequence_ranges(
            plan.cu_seqlens_cmp_k  # pyrefly: ignore [bad-argument-type]
        )
        width = max((end - start for start, end in block_ranges), default=0)
        indices = torch.full(
            (metadata.seq_len, width),
            -1,
            dtype=torch.int64,
            device=device,
        )
        for document_id, (q_start, q_end) in enumerate(_sequence_ranges(metadata.varlen.cu_seq_q)):
            c_start, c_end = block_ranges[document_id]
            compressed_len = c_end - c_start
            if compressed_len == 0:
                continue
            local = torch.arange(q_end - q_start, device=device)
            limit = torch.div(local + 1, self.compress_ratio, rounding_mode="floor")
            candidates = torch.arange(compressed_len, device=device).expand(q_end - q_start, -1)
            indices[q_start:q_end, :compressed_len] = torch.where(candidates < limit.unsqueeze(1), candidates, -1)
        return indices

    def _select_topk(
        self,
        idx_q,
        idx_k,
        idx_w,
        metadata: ReferenceCompressedVarlenMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CSA reference: FP32 indexer scores, per-document causal top-k.

        Returns document-local compressed indices ``[T, topk]`` (``-1`` for
        unused slots) and the masked scores, matching the baseline's
        formulation.
        """
        if idx_q is None or idx_k is None or idx_w is None:
            raise ValueError("ratio-4 golden reference requires all LI projection tensors.")
        plan = metadata.plans[4]
        idx_q = idx_q.flatten(0, 1)
        idx_k = idx_k.flatten(0, 1)[: plan.cu_seqlens_cmp_k[-1]]
        idx_w = idx_w.flatten(0, 1)
        total_tokens = idx_q.shape[0]
        total_blocks = idx_k.shape[0]

        index_score = torch.einsum("thd,ud->thu", idx_q.float(), idx_k.float())
        index_score = (index_score.relu_() * idx_w.float().unsqueeze(-1)).sum(dim=1)

        query_docs = metadata.reference.doc_of_token.flatten()
        token_positions = metadata.reference.pos_in_doc.flatten()
        causal_limit = torch.div(token_positions + 1, self.compress_ratio, rounding_mode="floor")
        block_docs = _packed_block_docs(plan, total_blocks, idx_q.device)
        if total_blocks:
            block_local = torch.arange(total_blocks, device=idx_q.device) - (
                plan.cu_seqlens_cmp_k[  # pyrefly: ignore [unsupported-operation]
                    block_docs
                ].long()
            )
            valid = torch.eq(query_docs.unsqueeze(1), block_docs.unsqueeze(0))
            valid = torch.logical_and(valid, torch.lt(block_local.unsqueeze(0), causal_limit.unsqueeze(1)))
            index_score = index_score.masked_fill(torch.logical_not(valid), torch.finfo(index_score.dtype).min)
        else:
            valid = torch.zeros((total_tokens, 0), dtype=torch.bool, device=idx_q.device)

        selected_width = min(self.index_topk, total_blocks)
        selected_score, selected_global = index_score.topk(selected_width, dim=-1)
        selected_valid = torch.gather(valid, dim=-1, index=selected_global)
        compressed_doc_starts = torch.searchsorted(block_docs, query_docs)
        selected_local = selected_global - compressed_doc_starts.unsqueeze(1)
        selected_local = torch.where(selected_valid, selected_local, -1)
        selected_score = selected_score.masked_fill(~selected_valid, float("-inf"))

        pad = self.index_topk - selected_width
        if pad:
            selected_local = F.pad(selected_local, (0, pad), value=-1)
            selected_score = F.pad(selected_score, (0, pad), value=float("-inf"))
        return selected_local, selected_score

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attn_sink=None,
        *,
        attention_masks: ReferenceCompressedVarlenMetadata | None = None,
    ):
        if not isinstance(attention_masks, CompressedVarlenMetadata):
            raise TypeError(
                "GoldenCompressedSparseInnerAttention requires "
                f"CompressedVarlenMetadata attention masks, got "
                f"{type(attention_masks)}."
            )
        metadata = attention_masks
        query = q.flatten(0, 1)
        original_kv = swa_k.flatten(0, 1)
        plan = metadata.plans.get(self.compress_ratio)
        compressed_kv = (
            query.new_empty((0, query.shape[-1]))
            if cmp_k is None or plan is None
            else cmp_k.flatten(0, 1)[: plan.cu_seqlens_cmp_k[-1]]
        )

        index_score = None
        if self.compress_ratio == 4:
            compressed_indices, _ = self._select_topk(idx_q, idx_k, idx_w, metadata)
        elif self.compress_ratio > 1:
            compressed_indices = self._packed_compressed_indices(metadata, query.device)
        else:
            compressed_indices = None

        outputs = []
        compressed = None if self.compress_ratio <= 1 else metadata.plans.get(self.compress_ratio)
        block_ranges = (
            None
            if compressed is None
            else _sequence_ranges(
                compressed.cu_seqlens_cmp_k  # pyrefly: ignore [bad-argument-type]
            )
        )
        for document_id, (q_start, q_end) in enumerate(_sequence_ranges(metadata.varlen.cu_seq_q)):
            length = q_end - q_start
            document_query = query[q_start:q_end].unsqueeze(0)
            document_kv = original_kv[q_start:q_end]
            indices = _window_topk_idxs(self.window_size, 1, length, query.device)
            if compressed is not None:
                c_start, c_end = block_ranges[  # pyrefly: ignore [unsupported-operation]
                    document_id
                ]
                document_compressed = compressed_kv[c_start:c_end]
                document_indices = compressed_indices[  # pyrefly: ignore [unsupported-operation]
                    q_start:q_end, : c_end - c_start
                ]
                document_indices = torch.where(  # pyrefly: ignore [no-matching-overload]
                    document_indices < 0,
                    document_indices,
                    document_indices + length,
                ).unsqueeze(0)
                indices = torch.cat([indices, document_indices], dim=-1)
                document_kv = torch.cat([document_kv, document_compressed], dim=0)
            result = _sparse_attn(
                document_query,
                document_kv.unsqueeze(0),
                attn_sink,  # pyrefly: ignore [bad-argument-type]
                indices,
                self.softmax_scale,
                return_lse=False,
            )
            outputs.append(result.squeeze(0))  # pyrefly: ignore [missing-attribute]

        output = torch.cat(outputs, dim=0).to(query.dtype)
        return output.reshape(metadata.batch_size, metadata.seq_len, *output.shape[1:])
