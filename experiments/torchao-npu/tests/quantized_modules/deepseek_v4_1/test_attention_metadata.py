# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The quantized attention's AI_CPU metadata, as the provider builds it from frames alone.

The selector is a hierarchy -- a pool source that publishes candidate blocks and the selectors
after it that search them -- so one geometry carries one QLI schedule (the candidate writer's width
is a launch argument, so a pool source and a pool-free selector share it) plus one QSLI schedule
and the count it is bound by.  All of them are AI_CPU operator outputs of the frames alone, and
QSLI's per-row candidate count is *derived* at the same time: QLI's own count only exists after the
forward has run, and a schedule built from it could not be precomputed at all.

What this file pins is that derivation (the kernels' arithmetic, not a bound: a count that is one
block too large or too small corrupts rows instead of failing) and the schedules' sizes and
arguments.  That the metadata rides the attention entry -- once per microbatch, rebuilt per rank,
pytree-transparent -- is the host config's contract, tested where that config lives.
"""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
from torchao_npu.quantized_modules.deepseek_v4_1.attention_metadata import (
    build_attention_metadata,
    candidate_sparse_length,
)

_GEOMETRY = dict(
    num_heads=64,
    head_dim=512,
    index_num_heads=2,
    index_head_dim=4,
    index_topk=32,
    window_size=128,
)
_POOL = dict(topk_blocks=3, block_size=8)


@pytest.fixture
def quant_ops(monkeypatch):
    """Every operator the quantized entry launches: count each launch, keep its arguments."""
    state = {"smla": [], "smla_grad": [], "slig": [], "qli": [], "qsli": []}

    def record(name, *args, **kwargs):
        out = torch.zeros(1, dtype=torch.int32)
        state[name].append((args, kwargs, out))
        return out

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla_metadata",
        lambda *a, **k: record("smla", *a, **k),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla_grad_metadata",
        lambda *a, **k: record("smla_grad", *a, **k),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad_metadata",
        lambda **k: record("slig", **k),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(
            quant_lightning_indexer_metadata=lambda **k: record("qli", **k),
            quant_sparse_lightning_indexer_metadata=lambda sparse_length, **k: record("qsli", sparse_length, **k),
        ),
        raising=False,
    )
    return state


def _frame(cu_seqlens: list[int], seqused: list[int], residual: list[int] | None = None):
    return SimpleNamespace(
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
        seqused=torch.tensor(seqused, dtype=torch.int32),
        residual=None if residual is None else torch.tensor(residual, dtype=torch.int32),
    )


def _reference(bounds, used_q, used_k, residual, *, ratio, topk_blocks, block_size) -> torch.Tensor:
    """The count row by row, straight from the length model: the arithmetic, not a bound.

    ``visible`` is the compressed-causal visible Key length of the row (a group counts once the
    query's own document has produced it), the count is its ceiling in blocks capped at the pool's
    capacity, and a row with nothing visible gets zero rather than one block.
    """
    counts = []
    for doc, (begin, end) in enumerate(pairwise(bounds)):
        span = end - begin
        rem = 0 if residual is None else residual[doc]
        for offset in range(span):
            if offset >= min(span, used_q[doc]):
                counts.append(0)
                continue
            visible = min(used_k[doc], max(0, (used_k[doc] * ratio + rem - used_q[doc] + offset + 1) // ratio))
            counts.append(0 if visible == 0 else min(topk_blocks, -(-visible // block_size)))
    return torch.tensor(counts, dtype=torch.int32).view(-1, 1)


@pytest.mark.parametrize(
    "bounds, used_q, used_k, residual, ratio, expected",
    [
        # The operator note's worked example: six queries over a 1024-token row at ratio 1, each
        # seeing 1019..1024 keys, which is 128 candidate blocks of eight positions.
        ([0, 6], [6], [1024], None, 1, [128] * 6),
        # Two documents of eight tokens, each a whole number of pooling groups.  The first query of
        # a document has produced no complete group, so it opens no block at all.
        ([0, 8, 16], [8, 8], [4, 4], [0, 0], 2, [0, 1, 1, 1, 1, 1, 1, 1] * 2),
        # A trailing partial group: five tokens are two groups and a remainder of one.
        ([0, 5], [5], [2], [1], 2, [0, 1, 1, 1, 1]),
        # Padding rows: addressed by the grid but never read, so they select nothing.
        ([0, 4], [2], [2], None, 1, [1, 1, 0, 0]),
    ],
    ids=["worked-example", "two-documents", "partial-group", "padding"],
)
def test_the_count_is_a_closed_form_over_the_lengths(quant_ops, bounds, used_q, used_k, residual, ratio, expected):
    """One row per packed query token, in the kernels' own integer arithmetic.

    Every case is built from lengths alone -- the frames the attention override hands over -- so nothing
    here needs QLI to have run, which is the whole reason the count is derived instead of read back
    off its candidate table.
    """
    del quant_ops
    compressed = [0]
    for length in used_k:
        compressed.append(compressed[-1] + length)

    count = candidate_sparse_length(
        ratio=ratio,
        q=_frame(bounds, used_q),
        cmp_k=_frame(compressed, used_k, residual),
        num_rows=bounds[-1],
        topk_blocks=2048,
        block_size=8,
    )
    want = _reference(bounds, used_q, used_k, residual, ratio=ratio, topk_blocks=2048, block_size=8)

    assert count.dtype is torch.int32
    assert count.shape == (bounds[-1], 1) and count.is_contiguous()
    torch.testing.assert_close(count, want, rtol=0, atol=0)
    assert count.flatten().tolist() == expected


@pytest.mark.parametrize(
    "visible, topk_blocks, expected",
    [
        # The partial trailing block is always retained: 8191 tokens are 1024 blocks, not 1023.
        (8191, 2048, 1024),
        (8, 2048, 1),
        # Zero stays zero: a row with no visible Key gets no block at all.
        (0, 2048, 0),
        # The cap applies to the block count, after the ceiling, and equals the pool's capacity.
        (600, 3, 3),
    ],
    ids=["ceiling", "one-block", "zero", "cap-after-ceiling"],
)
def test_the_three_rules_the_count_cannot_simplify(quant_ops, visible, topk_blocks, expected):
    """Ceiling, zero, then the cap: the three places a "safe" bound would desynchronise QSLI.

    An underestimate clips the last row's trailing tiles and an overestimate makes the schedule and
    the kernel disagree about which row is complete -- both silently, so the arithmetic is pinned
    rather than approximated.
    """
    del quant_ops
    # One query row that has just produced ``visible`` compressed entries: a document of
    # ``visible * block_size`` tokens read by its last query.
    tokens = max(visible, 1) * 8
    q = _frame([0, tokens], [tokens])
    cmp_k = _frame([0, visible], [visible], None)

    count = candidate_sparse_length(ratio=1, q=q, cmp_k=cmp_k, num_rows=tokens, topk_blocks=topk_blocks, block_size=8)

    assert count[-1].item() == expected


def test_the_entry_carries_the_pair_and_the_two_schedules(quant_ops):
    """One launch of each operator, and the search bound by the count it derived.

    The candidate writer's width is a launch argument, so a pool source and a selector that scores
    every entry share the one QLI schedule; QSLI's is sized by the count this frame set implies and
    is launched with exactly that tensor.
    """
    entry = build_attention_metadata(
        ratio=1,
        q=_frame([0, 8, 16], [8, 8]),
        swa_k=_frame([0, 8, 16], [8, 8]),
        cmp_k=_frame([0, 8, 16], [8, 8]),
        num_rows=16,
        **_GEOMETRY,
        **_POOL,
    )

    assert len(quant_ops["smla"]) == 1 and len(quant_ops["smla_grad"]) == 1
    assert len(quant_ops["slig"]) == 1 and len(quant_ops["qli"]) == 1 and len(quant_ops["qsli"]) == 1
    # The pair is the window's own geometry: head-major, with the selection width on the compressed
    # axis' top-k.
    args, kwargs, fwd = quant_ops["smla"][0]
    assert args == (64, 1, 512)
    assert kwargs["cmp_topk"] == 32 and kwargs["has_cmp_kv"] is True
    assert kwargs["ori_win_left"] == _GEOMETRY["window_size"] - 1
    assert entry.sparse_attn.fwd is fwd and entry.sparse_attn.bwd is quant_ops["smla_grad"][0][2]
    # No width is passed to the QLI metadata: the operator defaults to "no pool", and the width a
    # layer runs with is stated by that layer's launch.
    assert "candidate_topk_blocks" not in quant_ops["qli"][0][1]
    assert "candidate_block_size" not in quant_ops["qli"][0][1]
    # The search is bound by the derived count, byte for byte the tensor the entry carries.
    args, kwargs, built = quant_ops["qsli"][0]
    sparse_length = args[0]
    assert sparse_length is entry.lightning_indexer.sparse_length and built is entry.lightning_indexer.fwd_sparse
    assert kwargs["candidate_block_size"] == 8 and kwargs["quant_mode"] == 1
    assert sparse_length.shape == (16, 1) and sparse_length.dtype is torch.int32
    # Two documents of eight tokens at ratio 1: every row sees its own document's eight entries.
    assert sparse_length.flatten().tolist() == [1] * 16
    assert entry.lightning_indexer.fwd is quant_ops["qli"][0][2]
    assert entry.lightning_indexer.bwd is quant_ops["slig"][0][2]


def test_a_window_only_geometry_carries_the_pair_and_no_selector(quant_ops):
    """A geometry that pools at nothing runs no selector, so it has no schedules to carry."""
    entry = build_attention_metadata(
        ratio=0,
        q=_frame([0, 8], [8]),
        swa_k=_frame([0, 8], [8]),
        cmp_k=None,
        num_rows=8,
        **_GEOMETRY,
        **_POOL,
    )

    assert len(quant_ops["smla"]) == 1 and len(quant_ops["smla_grad"]) == 1
    assert quant_ops["qli"] == [] and quant_ops["qsli"] == [] and quant_ops["slig"] == []
    assert entry.lightning_indexer is None
    # A5 without a compressed axis addresses the window only.
    assert quant_ops["smla"][0][1]["cmp_mask_mode"] == 0


def test_a_geometry_without_a_pool_keeps_the_forward_schedule(quant_ops):
    """A pool-free stack still launches QLI, so it still needs its schedule; only the search goes."""
    entry = build_attention_metadata(
        ratio=1,
        q=_frame([0, 8], [8]),
        swa_k=_frame([0, 8], [8]),
        cmp_k=_frame([0, 8], [8]),
        num_rows=8,
        **{**_GEOMETRY, "topk_blocks": -1, "block_size": -1},
    )

    assert len(quant_ops["qli"]) == 1 and quant_ops["qsli"] == []
    assert entry.lightning_indexer.fwd is quant_ops["qli"][0][2]
    assert entry.lightning_indexer.fwd_sparse is None and entry.lightning_indexer.sparse_length is None
