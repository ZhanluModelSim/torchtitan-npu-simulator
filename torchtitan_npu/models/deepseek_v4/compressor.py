# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upstream compressor references and the NPU selector/reference helpers."""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
from attn_gym.sparse import lightning_indexer
from torchtitan.models.base.module import Module
from torchtitan.models.deepseek_v4 import compressor as v4_compressor
from torchtitan.models.deepseek_v4.compressor import (
    _hadamard,  # noqa: F401 -- preserve the existing helper import path
)

from torchtitan_npu.models.common.document_intervals import document_intervals


class Indexer(v4_compressor.Indexer):
    """Project indexer inputs and select the slots consumed by sparse attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(v4_compressor.Indexer.Config):
        inner_indexer: "InnerIndexer.Config"

    def __init__(self, config: Config):
        super().__init__(config)
        self.inner_indexer = config.inner_indexer.build()

    def forward(self, x, qr, *, positions, cu_seqlens=None, attention_metadata=None):
        idx_q, idx_k, idx_w = super().forward(x, qr, positions=positions, cu_seqlens=cu_seqlens)
        return self.inner_indexer(idx_q, idx_k, idx_w, attention_metadata=attention_metadata)


def select_compressed_slots(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """The reference selection: Attention Gym's packed ``lightning_indexer``.

    The operator applies each query's own document's causal limit itself and numbers the picks
    inside that document -- the numbering ``gather_attn`` reads -- so nothing rebases them here.
    Every row it cannot fill comes back as ``-1``, and the packed gather only reads entries
    inside the query's own document pool, so an empty pool can never name its neighbour's row.
    Its score is
    ``sum_h w * relu(q . k) / sqrt(H * D)``, and the caller's weights already carry the model's
    own scale, so multiplying them by ``sqrt(H * D)`` cancels that division exactly.  ``topk``
    is clamped to the pool's width, as this model's selection has always been.
    """
    _, heads, head_dim = idx_q.shape
    return lightning_indexer(
        idx_q.float().unsqueeze(0),
        idx_k.float().unsqueeze(0),
        idx_w.float().unsqueeze(0) * math.sqrt(heads * head_dim),
        min(topk, idx_k.size(0)),
        causal=True,
        compress_ratio=4,
        cu_seqlens=cu_seqlens.contiguous(),
        cu_seqlens_k=cu_seqlens_k.contiguous(),
        impl="reference",
    ).squeeze(0)


def student_logits_at_picks(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """The per-head scores at a document-local selection: the pass that carries gradient.

    The selection itself is discrete, so only the picks can be scored differentiably -- and a
    selection that returns indices alone leaves nothing to gather them from, which is why this
    is a pass of its own rather than a by-product of the selection.  ``topk_indices`` is
    renumbered onto the row-wide entries the pooled keys are stored in; a ``-1`` slot is
    gathered harmlessly and marked ``-inf``, the value a softmax drops. The reference path
    requires ``B == 1``, so the key axis is squeezed. Returns ``[1, L, K]``, head-summed and in
    the model's own score scale (``select_compressed_slots`` cancels the operator's internal
    division; this pass never divides).

    This is the pass ``InnerIndexer`` returns beside the selection, and the fused
    selector's backward consumes the same quantity: at the picks, the student logits are the
    only differentiable edge of the selection.
    """
    pool_start_T1, _, _ = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=idx_q.size(0))
    picked_TKDi = idx_k[(topk_indices + pool_start_T1).clamp_min(0)]
    scores_THK = torch.einsum("thd,tkd->thk", idx_q, picked_TKDi)
    weighted_THK = scores_THK.relu() * idx_w.unsqueeze(-1)
    return weighted_THK.sum(dim=1).masked_fill(topk_indices < 0, -torch.inf)


class InnerIndexer(Module):
    requires_teacher = False

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int

    def __init__(self, config: Config):
        super().__init__()
        self.index_topk = config.index_topk
        self.return_scores = True

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        """Select the compressed slots and score them differentiably.

        Returns document-local indices and the corresponding student logits. Native CSA can
        skip the logits when its auxiliary objective is disabled.
        """
        # Native TorchTitan metadata includes the longest document for the CSA top-k bound.
        documents = getattr(attention_metadata, "documents", attention_metadata)
        if documents is not None or attention_metadata is None:
            with spmd.no_typecheck():
                return v4_compressor.Indexer.select(
                    idx_q,
                    idx_k,
                    idx_w,
                    max_seqlen=idx_q.size(0) if documents is None else documents.max_k,
                    ratio=4,
                    topk=self.index_topk,
                    cu_seqlens=None if documents is None else documents.cu_seq_q,
                    return_scores=self.training and self.return_scores,
                )
        if attention_metadata.ratio <= 1:
            raise ValueError("InnerIndexer requires a compressed geometry (ratio > 1).")
        cu_seqlens = attention_metadata.q.cu_seqlens
        cmp_frame = attention_metadata.cmp_k
        if cmp_frame is None:
            raise ValueError("a compressed geometry needs the entry's compressed frame")
        cu_seqlens_k = cmp_frame.cu_seqlens
        # Selection is discrete: it carries no gradient.  The operator's order is kept as-is
        # for the reference path; the fused selector emits the canonical descending order
        # (see the module docstring), so the two paths agree on the *set* and on ``-1``.
        with torch.no_grad():
            topk_indices = select_compressed_slots(
                idx_q,
                idx_k,
                idx_w,
                cu_seqlens,
                cu_seqlens_k,
                self.index_topk,
            )
        topk_scores = student_logits_at_picks(
            idx_q,
            idx_k,
            idx_w,
            topk_indices,
            cu_seqlens,
            cu_seqlens_k,
        )
        return topk_indices, topk_scores
