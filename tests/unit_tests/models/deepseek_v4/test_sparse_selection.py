# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The selector's two halves: the packed selection, and the rescore that carries gradient.

The selection is discrete, so its picks are an op's output; the student logits the indexer is
trained through have to be recomputed at those picks, in the row-wide numbering the pooled keys
are stored in.  These cases pin that mapping and the marking of a padded slot.
"""

import pytest
import torch
from torchtitan.models.common.attention import VarlenMetadata

pytestmark = pytest.mark.usefixtures("dsv4")

DIM = 8
_RATIO = 4
_DOC = 8  # three documents of eight tokens: two compressed slots each
_CU = torch.tensor([0, _DOC, 2 * _DOC, 3 * _DOC], dtype=torch.int32)
# Per row, the picks in the selection's own numbering; the middle document pads its second slot.
_PICKS = torch.tensor([[0, 1]] * _DOC + [[0, -1]] * _DOC + [[1, -1]] * _DOC, dtype=torch.int64).view(1, -1, 2)


def _metadata(dsv4):
    varlen = VarlenMetadata(cu_seq_q=_CU, cu_seq_k=_CU.clone(), max_q=_DOC, max_k=_DOC)
    return dsv4.metadata.build_compressed_varlen_metadata(varlen, (_RATIO,))


def _inputs(seqlen, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(1, seqlen, 2, DIM, dtype=torch.float64),
        torch.randn(1, seqlen // _RATIO, DIM, dtype=torch.float64),
        torch.randn(1, seqlen, 2, dtype=torch.float64),
    )


def _rescore(dsv4, metadata, idx_q, idx_k, idx_w, picks):
    return dsv4.compressor.student_logits_at_picks(
        idx_q, idx_k, idx_w, picks, metadata.varlen.cu_seq_q, metadata.plans[_RATIO].cu_seqlens_cmp_k
    )


def test_student_logits_rescore_a_document_local_selection(dsv4):
    """Each pick is read as its own document's entry, not as a row-wide one."""
    metadata = _metadata(dsv4)
    seqlen = int(_CU[-1])
    starts = metadata.plans[_RATIO].cu_seqlens_cmp_k
    idx_q, idx_k, idx_w = _inputs(seqlen)

    scores = _rescore(dsv4, metadata, idx_q, idx_k, idx_w, _PICKS)

    expected = torch.full((1, seqlen, 2), -torch.inf, dtype=torch.float64)
    for token in range(seqlen):
        start = int(starts[token // _DOC])
        for slot, local in enumerate(_PICKS[0, token].tolist()):
            if local < 0:
                continue
            dot = torch.einsum("hd,d->h", idx_q[0, token], idx_k[0, start + local])
            expected[0, token, slot] = (dot.relu() * idx_w[0, token]).sum()

    torch.testing.assert_close(scores, expected, rtol=1e-12, atol=1e-12)
    # The marking agrees with the selection slot for slot: only -1 is un-scored.
    assert torch.equal(torch.isfinite(scores), _PICKS >= 0)
    # Reading the same pick row-wide would score document 0's entry instead.
    row_wide = (torch.einsum("hd,d->h", idx_q[0, _DOC], idx_k[0, 0]).relu() * idx_w[0, _DOC]).sum()
    assert not torch.isclose(scores[0, _DOC, 0], row_wide)


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
    picks[:, :_DOC] = 0
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
