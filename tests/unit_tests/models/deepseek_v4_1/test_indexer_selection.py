# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The selector's ``-1`` padding: what those slots score, and where the gradient lands.

The pool is numbered row-wide while the selection is document-local, so a query can reach
fewer entries than ``index_topk`` -- a one-entry document always leaves ``-1`` slots behind.
Those slots must score exactly ``-inf`` and must carry neither a value nor a gradient, or a
later change could hand a query another document's entry.
"""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.indexer import FULL, REINDEX, InnerIndexer

# Three documents of 4, 8 and 2 tokens; at ratio 2 they pool 2, 4 and 1 entries.
BOUNDS = [0, 4, 12, 14]
POOL = [0, 2, 6, 7]
HEADS, HEAD_DIM, TOPK = 2, 4, 3
# The pool start of each token's own document, straight from BOUNDS/POOL.
POOL_START_PER_TOKEN = [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 6, 6]

# Hand-computed per token: the document-local slots the selection emits.  Visibility is
# ``[pool_start, pool_start + (local + 1) // ratio)``, the top-k keeps the highest-valued
# entries of it, and the emission is position-descending with ``-1`` at the tail.
EXPECTED_LOCALS = [
    [-1, -1, -1],
    [0, -1, -1],
    [0, -1, -1],
    [1, 0, -1],  # document 1: 4 tokens, 2 entries
    [-1, -1, -1],
    [0, -1, -1],
    [0, -1, -1],
    [1, 0, -1],
    [1, 0, -1],
    [2, 1, 0],
    [2, 1, 0],
    [3, 2, 1],  # document 2: 8 tokens, 4 entries
    [-1, -1, -1],
    [0, -1, -1],  # document 3: 2 tokens, 1 entry
]


def _masks():
    """The row's own metadata, which is all the reference selector reads.

    The pool's grid is this layer's ratio applied to those boundaries, so the selector needs
    no separate frame: ``POOL`` stays a test-side statement of what the grid should be.  The
    backend that built the row folded that same grid into ``cu_seq_k``, which is pinned here
    to be the selector's own division -- one set of boundaries, however it is read.
    """
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(BOUNDS)])
    config = CompressedSparseInnerAttention2.Config(window_size=8, softmax_scale=0.5, compress_ratio=2)
    metadata = config.build_attention_metadata(positions)
    torch.testing.assert_close(metadata.cu_seq_k, metadata.cu_seq_q // 2, rtol=0, atol=0)
    assert metadata.cu_seq_k.tolist() == POOL
    return metadata


def _selector():
    return InnerIndexer(
        InnerIndexer.Config(
            mode=FULL,
            compress_ratio=2,
            num_index_heads=HEADS,
            index_head_dim=HEAD_DIM,
            index_topk=TOPK,
        )
    )


def _values():
    """One per pool entry, distinct so the top-k has no ties."""
    return torch.arange(1, POOL[-1] + 1, dtype=torch.float32)


def _inputs(*, token: int | None = None):
    """One-hot queries and constant key rows: a slot's score is ``heads * value``.

    ``idx_q[..., h, :]`` is the h-th unit vector and each ``idx_k`` row repeats its value, so
    ``sum_h relu(q . k) * w`` is ``heads * value`` for the entry the slot names.  ``token``
    keeps a single query position alive, which is how the gradient test isolates one document.
    """
    values = _values()
    idx_k = values[:, None].repeat(1, HEAD_DIM)
    positions = range(BOUNDS[-1]) if token is None else (token,)
    idx_q = torch.zeros(BOUNDS[-1], HEADS, HEAD_DIM)
    for position in positions:
        for head in range(HEADS):
            idx_q[position, head, head] = 1.0
    return values, idx_k, idx_q, torch.ones(BOUNDS[-1], HEADS)


def test_unused_slots_are_minus_one_and_score_minus_inf():
    values, idx_k, idx_q, weights = _inputs()
    indices, scores, _ = _selector()(idx_q, idx_k, weights, _masks(), candidates_BL1C=None)

    torch.testing.assert_close(indices, torch.tensor(EXPECTED_LOCALS, dtype=indices.dtype), rtol=0, atol=0)

    expected = torch.full((BOUNDS[-1], TOPK), -torch.inf)
    for position, slots in enumerate(EXPECTED_LOCALS):
        for slot, local in enumerate(slots):
            if local >= 0:
                expected[position, slot] = HEADS * values[local + POOL_START_PER_TOKEN[position]]
    torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-6)


def test_gradient_reaches_only_the_selected_entries_of_the_own_document():
    """Only document 1's last token takes part, so only its two entries may receive gradient.

    Its selection is ``[1, 0, -1]``: the ``-1`` slot is gathered from the clamped row-wide
    position 0 and then masked to ``-inf``, which must leave no gradient of its own -- and
    nothing at all may reach documents 2 and 3.
    """
    values, idx_k, idx_q, weights = _inputs(token=3)
    idx_k = idx_k.clone().requires_grad_(True)
    idx_q = idx_q.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)

    indices, scores, _ = _selector()(idx_q, idx_k, weights, _masks(), candidates_BL1C=None)
    torch.testing.assert_close(indices[3], torch.tensor([1, 0, -1], dtype=indices.dtype), rtol=0, atol=0)

    scores.sum().backward()
    touched = idx_k.grad.abs().sum(dim=1) > 0
    expected = torch.zeros(values.numel(), dtype=torch.bool)
    expected[0] = expected[1] = True
    assert torch.equal(touched, expected), f"gradient reached entries {touched.nonzero().flatten().tolist()}"


@pytest.mark.parametrize(
    ("candidate_count", "packed"),
    [(25, False), (32, False), (None, False), (25, True)],
    ids=["underfull-original-k32", "exact-original-k32", "no-candidate-pool", "packed-boundary"],
)
def test_reindex_preserves_frozen_selection_semantics(candidate_count, packed):
    """Migration preserves dabeb29's visible-slot rule, including its known pool-padding bug."""
    length = 128 if packed else 64
    boundaries = [0, 64, 128] if packed else [0, 64]
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(boundaries)])
    metadata = CompressedSparseInnerAttention2.Config(
        window_size=8, softmax_scale=0.5, compress_ratio=1
    ).build_attention_metadata(positions)
    selector = InnerIndexer(
        InnerIndexer.Config(
            mode=REINDEX,
            compress_ratio=1,
            num_index_heads=2,
            index_head_dim=4,
            index_topk=32,
            candidate_topk_blocks=4 if candidate_count is not None else -1,
            candidate_block_size=8 if candidate_count is not None else -1,
        )
    )
    values = torch.arange(1, length + 1, dtype=torch.float32)
    keys = values[:, None].expand(-1, 4)
    queries = torch.zeros(length, 2, 4)
    queries[:, 0, 0] = queries[:, 1, 1] = 1
    weights = torch.ones(length, 2)
    candidates = torch.zeros(length, length, dtype=torch.bool) if candidate_count is not None else None
    visible = torch.zeros(length, length, dtype=torch.bool)
    pool_start = torch.zeros(length, 1, dtype=torch.int64)
    for begin, end in pairwise(boundaries):
        for token in range(begin, end):
            visible[token, begin : token + 1] = True
            pool_start[token] = begin
            if candidates is not None:
                candidates[token, begin : begin + candidate_count] = True
    # Frozen [1,T,N] oracle: the score pass respects candidates, while the
    # original post-topk validity rule reads visibility alone.
    old_scores = torch.stack([values * 2 for _ in range(length)]).unsqueeze(0)
    old_scores = old_scores.masked_fill(~visible.unsqueeze(0), -torch.inf)
    if candidates is not None:
        old_scores = old_scores.masked_fill(~candidates.unsqueeze(0), -torch.inf)
    old_picks = old_scores.topk(32, dim=-1, sorted=False).indices.squeeze(0)
    expected = torch.where(visible.gather(-1, old_picks), old_picks - pool_start, -1).sort(-1, descending=True).values
    expected_scores = torch.full((length, 32), -torch.inf)
    for token in range(length):
        for slot, index in enumerate(expected[token]):
            if index >= 0:
                expected_scores[token, slot] = 2 * values[int(pool_start[token]) + int(index)]

    indices, scores, returned = selector(queries, keys, weights, metadata, candidates_BL1C=candidates)

    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    assert returned is candidates
    if candidate_count == 25 and not packed:
        assert int((indices[-1] >= 0).sum()) == 32
        assert int(candidates[-1].sum()) == 25
