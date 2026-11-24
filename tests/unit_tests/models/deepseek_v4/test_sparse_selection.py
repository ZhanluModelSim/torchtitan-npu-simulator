# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The selector's two halves: the packed selection, and the rescore that carries gradient.

The selection is discrete, so its picks are an op's output; the student logits the indexer is
trained through have to be recomputed at those picks, in the row-wide numbering the pooled keys
are stored in.  These cases pin that mapping and the marking of a padded slot.
"""

from itertools import pairwise

import pytest
import torch

pytestmark = pytest.mark.usefixtures("dsv4")

DIM = 8
_RATIO = 4
_DOC = 8  # three documents of eight tokens: two compressed slots each
_CU = torch.tensor([0, _DOC, 2 * _DOC, 3 * _DOC], dtype=torch.int32)
# Per row, the picks in the selection's own numbering; the middle document pads its second slot.
_PICKS = torch.tensor([[0, 1]] * _DOC + [[0, -1]] * _DOC + [[1, -1]] * _DOC, dtype=torch.int64).view(-1, 2)


def _metadata(dsv4):
    positions = torch.cat([torch.arange(end - start) for start, end in pairwise(_CU.tolist())])
    return dsv4.metadata.build_v4_metadata(positions, _RATIO, cu_seq_q=_CU)


def _inputs(seqlen, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(seqlen, 2, DIM, dtype=torch.float64),
        torch.randn(seqlen // _RATIO, DIM, dtype=torch.float64),
        torch.randn(seqlen, 2, dtype=torch.float64),
    )


def _rescore(dsv4, metadata, idx_q, idx_k, idx_w, picks):
    return dsv4.compressor.student_logits_at_picks(
        idx_q, idx_k, idx_w, picks, metadata.swa_k.cu_seqlens, metadata.cmp_k.cu_seqlens
    )


def test_student_logits_rescore_a_document_local_selection(dsv4):
    """Each pick is read as its own document's entry, not as a row-wide one."""
    metadata = _metadata(dsv4)
    seqlen = int(_CU[-1])
    starts = metadata.cmp_k.cu_seqlens
    idx_q, idx_k, idx_w = _inputs(seqlen)

    scores = _rescore(dsv4, metadata, idx_q, idx_k, idx_w, _PICKS)

    expected = torch.full((seqlen, 2), -torch.inf, dtype=torch.float64)
    for token in range(seqlen):
        start = int(starts[token // _DOC])
        for slot, local in enumerate(_PICKS[token].tolist()):
            if local < 0:
                continue
            dot = torch.einsum("hd,d->h", idx_q[token], idx_k[start + local])
            expected[token, slot] = (dot.relu() * idx_w[token]).sum()

    torch.testing.assert_close(scores, expected, rtol=1e-12, atol=1e-12)
    # The marking agrees with the selection slot for slot: only -1 is un-scored.
    assert torch.equal(torch.isfinite(scores), _PICKS >= 0)
    # Reading the same pick row-wide would score document 0's entry instead.
    row_wide = (torch.einsum("hd,d->h", idx_q[_DOC], idx_k[0]).relu() * idx_w[_DOC]).sum()
    assert not torch.isclose(scores[_DOC, 0], row_wide)


def test_the_rescore_carries_the_indexer_gradient(dsv4):
    """The pass exists to be differentiated: the indexer's projections get real gradients."""
    metadata = _metadata(dsv4)
    seqlen = int(_CU[-1])
    idx_q, idx_k, idx_w = (tensor.clone().requires_grad_(True) for tensor in _inputs(seqlen))

    scores = _rescore(dsv4, metadata, idx_q, idx_k, idx_w, _PICKS)
    scores.sum().backward()

    for tensor in (idx_q, idx_k, idx_w):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert float(idx_w.grad.abs().sum()) > 0


def test_a_document_without_a_pick_gets_no_indexer_gradient(dsv4):
    """Every slot of a padded document is un-scored, and that edge carries no gradient.

    The marking is what cuts it: the row's scores are ``-inf``, so the backward of the sum
    reaches that edge with a zero and the queries of a document that selected nothing are
    never trained.
    """
    metadata = _metadata(dsv4)
    seqlen = int(_CU[-1])
    picks = torch.full_like(_PICKS, -1)
    picks[:_DOC] = 0
    idx_q, idx_k, idx_w = (tensor.clone().requires_grad_(True) for tensor in _inputs(seqlen))

    _rescore(dsv4, metadata, idx_q, idx_k, idx_w, picks).sum().backward()

    assert float(idx_q.grad[0, :_DOC].abs().sum()) > 0
    assert float(idx_q.grad[0, _DOC:].abs().sum()) == 0
    assert float(idx_w.grad[0, _DOC:].abs().sum()) == 0


def test_an_unscored_slot_is_marked_with_the_value_softmax_drops(dsv4):
    metadata = _metadata(dsv4)
    seqlen = int(_CU[-1])
    scores = _rescore(dsv4, metadata, *_inputs(seqlen), _PICKS)
    assert torch.isinf(scores[~(_PICKS >= 0)]).all()
    assert (scores[~(_PICKS >= 0)] < 0).all()


def test_lightning_indexer_returns_the_selection_and_its_scores(dsv4):
    """The indexer's forward is the selector's two outputs: picks and differentiable scores.

    The reference picks keep the selection operator's order (re-sorting would perturb the
    reference golden's summation order); they are document-local with ``-1`` for an unused
    slot.  The scores are the student logits at exactly those picks, ``-inf`` on padding,
    and they carry gradient to the indexer's projections.
    """
    metadata = _metadata(dsv4)
    seqlen = int(_CU[-1])
    idx_q, idx_k, idx_w = (tensor.clone().requires_grad_(True) for tensor in _inputs(seqlen))
    module = dsv4.compressor.InnerIndexer(dsv4.compressor.InnerIndexer.Config(index_topk=2))

    picks, scores = module(idx_q, idx_k, idx_w, attention_metadata=metadata)

    assert picks.shape == (seqlen, 2) and picks.dtype == torch.int32
    assert scores.shape == picks.shape
    # The reference emits the selection operator's order; only the set and the ``-1``
    # marking are part of the shared contract.
    expected_picks = dsv4.compressor.select_compressed_slots(
        idx_q.detach(),
        idx_k.detach(),
        idx_w.detach(),
        metadata.swa_k.cu_seqlens,
        metadata.cmp_k.cu_seqlens,
        2,
    )
    assert torch.equal(picks, expected_picks)
    # The scores are the same rescore pass, on exactly the emitted slots.
    expected = _rescore(dsv4, metadata, idx_q.detach(), idx_k.detach(), idx_w.detach(), picks)
    torch.testing.assert_close(scores, expected, rtol=1e-12, atol=1e-12)
    assert torch.isinf(scores[picks < 0]).all()
    assert (scores[picks < 0] < 0).all()

    # The returned scores are the differentiable edge the indexer is trained through.
    torch.nan_to_num(scores, neginf=0.0).sum().backward()
    for tensor in (idx_q, idx_k, idx_w):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert float(idx_w.grad.abs().sum()) > 0


def test_lightning_indexer_refuses_a_geometry_without_a_compressed_pool(dsv4):
    positions = torch.cat([torch.arange(end - start) for start, end in pairwise(_CU.tolist())])
    metadata = dsv4.metadata.build_v4_metadata(positions, 1)
    module = dsv4.compressor.InnerIndexer(dsv4.compressor.InnerIndexer.Config(index_topk=2))
    idx_q, idx_k, idx_w = _inputs(int(_CU[-1]))
    with pytest.raises(ValueError, match="compressed"):
        module(idx_q, idx_k, idx_w, attention_metadata=metadata)
