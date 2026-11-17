# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The sparse core's own contract: the slots it addresses, and the exit it reserves.

The core reads only the packed offsets the metadata carries, so these cases run it from the
common metadata and spell the rules out again from the plan.  Its numerics against the eager
oracle live in ``test_dsv4.py``.
"""

import pytest
import torch
from torchtitan.models.common.attention import VarlenMetadata

pytestmark = pytest.mark.usefixtures("dsv4")

HD = 16
_SHORT = torch.tensor([0, 10, 27, 47, 64], dtype=torch.int32)
_LONG = torch.tensor([0, 320, 640], dtype=torch.int32)
_RATIOS = (1, 4, 128)
_WINDOW, _SCALE, _TOPK = 8, 0.25, 8


def _metadata(dsv4, cu):
    longest = int((cu[1:] - cu[:-1]).max())
    varlen = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=longest, max_k=longest)
    return dsv4.metadata.build_compressed_varlen_metadata(varlen, _RATIOS)


def _core(dsv4, ratio, *, return_lse=False):
    return dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=_WINDOW,
            compress_ratio=ratio,
            softmax_scale=_SCALE,
            index_topk=_TOPK,
            return_lse=return_lse,
        )
    )


def _inputs(ratio, cu, seed=0):
    """Query, window keys, compressed pool, and the sink."""
    torch.manual_seed(seed)
    seqlen, heads = int(cu[-1]), 3
    n_cmp = 0 if ratio == 1 else seqlen // ratio
    q = torch.randn(1, seqlen, heads, HD)
    swa_k = torch.randn(1, seqlen, HD)
    cmp_k = None if n_cmp == 0 else torch.randn(1, n_cmp, HD)
    return q, swa_k, cmp_k, torch.randn(heads)


def _visible_local_slots(dsv4, metadata, ratio, seqlen, n_cmp):
    """The HCA rule spelled out again: own document, whole groups, causally complete."""
    starts = metadata.plans[ratio].cu_seqlens_cmp_k
    documents = torch.searchsorted(metadata.varlen.cu_seq_q[1:], torch.arange(seqlen), right=True)
    doc_start = starts[documents]
    local = torch.arange(seqlen) - metadata.varlen.cu_seq_q[documents]
    ends = torch.minimum(starts[documents + 1], doc_start + (local + 1) // ratio)
    slots = torch.arange(n_cmp).unsqueeze(0)
    visible = (slots >= doc_start.unsqueeze(1)) & (slots < ends.unsqueeze(1))
    return torch.where(visible, slots.expand(seqlen, -1) - doc_start.unsqueeze(1), -1).unsqueeze(0)


def test_the_core_reads_only_the_packed_offsets(dsv4):
    """No reference tier and no block mask: the common metadata is the whole input."""
    common = _metadata(dsv4, _SHORT)
    assert not hasattr(common, "reference")
    seqlen = int(_SHORT[-1])
    q, swa_k, cmp_k, sink = _inputs(4, _SHORT)
    torch.manual_seed(2)
    picks = dsv4.compressor.select_compressed_slots(
        torch.randn(1, seqlen, 4, 8),
        torch.randn(1, seqlen // 4, 8),
        torch.randn(1, seqlen, 4),
        common.varlen.cu_seq_q,
        common.plans[4].cu_seqlens_cmp_k,
        _TOPK,
    )

    with torch.no_grad():
        out = _core(dsv4, 4)(q, swa_k, cmp_k, attn_sink=sink, sparse_indices=picks, attention_masks=common)

    assert out.shape == (1, seqlen, 3, HD) and torch.isfinite(out).all()


def test_hca_slots_stay_inside_the_query_document(dsv4):
    """Ratio 128 reaches its whole visible prefix, numbered inside its own document."""
    cu = _LONG
    metadata = _metadata(dsv4, cu)
    seqlen, ratio = int(cu[-1]), 128
    n_cmp = seqlen // ratio

    local = _core(dsv4, ratio)._slots(
        None, metadata.varlen.cu_seq_q, metadata.plans[ratio].cu_seqlens_cmp_k, seqlen, n_cmp
    )

    assert torch.equal(local, _visible_local_slots(dsv4, metadata, ratio, seqlen, n_cmp))


def test_the_teacher_exit_is_reserved_and_inert(dsv4):
    """``return_lse`` hands back the per-head log-sum-exp and changes nothing else."""
    cu = _LONG
    metadata = _metadata(dsv4, cu)
    seqlen = int(cu[-1])
    q, swa_k, cmp_k, sink = _inputs(128, cu)

    with torch.no_grad():
        plain = _core(dsv4, 128)(q, swa_k, cmp_k, attn_sink=sink, attention_masks=metadata)
        with_lse, lse = _core(dsv4, 128, return_lse=True)(q, swa_k, cmp_k, attn_sink=sink, attention_masks=metadata)

    assert torch.equal(plain, with_lse)
    assert lse.shape == (1, 3, seqlen) and torch.isfinite(lse).all()
