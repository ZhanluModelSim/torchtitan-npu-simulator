# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused sparse adapter's coordinate translation and gradient plumbing."""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.override.deepseek_v4_1.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("bounds", [[0, 8], [0, 4, 8]], ids=["single-doc", "two-docs"])
def test_kernel_receives_document_local_indices(monkeypatch, v41_metadata, ratio, bounds):
    """The kernel gets the selection as the selector made it: document-local, in TND layout.

    Nothing about the *values* is translated -- the fused selector and this port share one
    coordinate system, and the port leaves the order alone because the teacher's slot
    positions follow it.  The layout is translated: the model carries ``[B, L, 1, K]`` and
    a TND kernel wants ``[T, N2, K]``.
    """
    length = bounds[-1]
    # The entry this layer would read in a run: its own ratio's frames and, for the selection's
    # width, its own geometry's metadata -- this test replaces the kernel node wholesale, so what
    # it pins is the layout the node is handed, but the metadata is still the real pairing.
    metadata = v41_metadata(
        bounds,
        ratio=ratio,
        geometry=AicpuGeometry(
            window_size=2, num_heads=2, head_dim=4, index_topk=5, index_num_heads=2, index_head_dim=4
        ),
    )
    assert torch.equal(metadata.cu_seq_q, torch.tensor(bounds, dtype=torch.int32))
    cu_cmp = metadata.cmp_k.cu_seqlens

    # Document-local indices, -1 for unused slots, interleaved: another document's first
    # entry, the first/last own entry, an unused slot and an exclusive-end index -- the
    # last two are a document's own first entries repeated, so they are inert either way.
    indices = torch.empty((1, length, 5), dtype=torch.int32)
    for doc, (begin, end) in enumerate(pairwise(bounds)):
        start, stop = int(cu_cmp[doc]), int(cu_cmp[doc + 1])
        own_last = stop - start - 1
        indices[0, begin:end] = torch.tensor([0, -1, own_last, stop - start, stop])

    calls = []

    def kernel(q, original, shared, selection, *args):
        calls.append(selection)
        return torch.zeros_like(q), torch.zeros((1, q.shape[0], q.shape[1]))

    monkeypatch.setattr(ascendc._SparseMLA, "apply", kernel)
    attention = ascendc.AscV41SparseAttention.Config(
        window_size=2,
        softmax_scale=0.5,
        compress_ratio=ratio,
        # The kernel node is replaced below, so this test is about the selection's layout rather
        # than about the metadata lookup; the config still states the geometry the metadata above
        # was sized with, the way a run pairs them.
        num_heads=2,
        head_dim=4,
        index_topk=5,
        index_num_heads=2,
        index_head_dim=4,
    ).build()
    q = torch.zeros((1, length, 2, 4), dtype=torch.bfloat16)
    topk_before = indices.clone()
    out = attention(
        q,
        torch.zeros((1, length, 4), dtype=torch.bfloat16),
        torch.zeros(2),
        metadata,
        cmp_k=torch.zeros((1, int(cu_cmp[-1]), 4), dtype=torch.bfloat16),
        topk_indices=indices,
        # A training-mode layer with a compressed pool carries the teacher on this
        # tensor's gradient, so the port requires the selection's student logits.
        topk_scores=torch.zeros((1, length, indices.shape[-1]), dtype=torch.float32),
    )
    assert out.shape == q.shape
    assert len(calls) == 1
    # The kernel receives the selection verbatim: this port shares the selector's
    # coordinate system and does not translate or reorder it, so ``-1`` stays where the
    # selector left it.  A reorder here would permute the indices without permuting the
    # logits the teacher is scattered back onto.
    # ``[B, L, 1, K]`` in, ``[T, 1, K]`` at the kernel.
    torch.testing.assert_close(calls[0], indices.reshape(-1, 1, indices.shape[-1]), rtol=0, atol=0)
    # ... and never rewrites the caller's tensor in place.
    torch.testing.assert_close(indices, topk_before, rtol=0, atol=0)


@pytest.mark.parametrize("with_shared", [False, True], ids=["window", "shared-kv"])
def test_backward_routes_kernel_gradients_and_ignores_the_lse_gradient(monkeypatch, v41_metadata, with_shared):
    q = torch.ones(4, 2, 4, requires_grad=True)
    original = torch.ones(4, 1, 4, requires_grad=True)
    shared = torch.ones(2, 1, 4, requires_grad=True) if with_shared else None
    sink = torch.ones(2, requires_grad=True)
    indices = torch.zeros(4, 1, 2, dtype=torch.int32) if with_shared else None
    cu = torch.tensor([0, 4], dtype=torch.int32)
    cmp_cu = torch.tensor([0, 2], dtype=torch.int32) if with_shared else None
    remainder = torch.zeros(1, dtype=torch.int32) if with_shared else None
    meta, lse = torch.tensor([7]), torch.zeros(4, 2)
    forward_args = {}
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", lambda *a, **kw: meta)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", lambda *a, **kw: meta)

    def forward(q, **kwargs):
        forward_args.update(kwargs)
        return q.clone(), lse

    def backward(q_in, grad, output, saved_lse, **kwargs):
        assert q_in is q and saved_lse is lse
        torch.testing.assert_close(grad, torch.ones_like(q))
        for name in (
            "ori_kv",
            "cmp_kv",
            "cmp_sparse_indices",
            "sinks",
            "metadata",
            "cu_seqlens_q",
            "cu_seqlens_ori_kv",
            "cu_seqlens_cmp_kv",
            "cmp_residual_kv",
        ):
            assert kwargs[name] is forward_args[name]
        assert kwargs["softmax_scale"] == 0.5
        assert kwargs["cmp_ratio"] == (2 if with_shared else 1)
        assert kwargs["ori_win_left"] == 1
        return (
            torch.full_like(q, 2),
            torch.full_like(original, 3),
            torch.full_like(shared, 5) if with_shared else None,
            torch.full_like(sink, 7),
            None,
            None,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", forward)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)
    # The boundary triples are no longer arguments: they come out of the metadata.
    out, returned_lse = ascendc._SparseMLA.apply(
        q,
        original,
        shared,
        indices,
        sink,
        # The entry this layer reads: the geometry the ``apply`` signature is driven with below
        # (two heads of width four, a window of two, and a two-wide selection).
        v41_metadata(
            [0, 4],
            ratio=2 if with_shared else 0,
            geometry=AicpuGeometry(
                window_size=2,
                num_heads=2,
                head_dim=4,
                index_topk=2,
                index_num_heads=2,
                index_head_dim=4,
            ),
        ),
        0.5,
        2 if with_shared else 0,
        2,
        None,  # topk_scores: no teacher carrier in this test
        False,  # wants_teacher
    )
    # The LSE passes through as a detached teacher signal.
    assert returned_lse is lse
    assert not returned_lse.requires_grad
    out.sum().backward()
    for value, expected in ((q, 2), (original, 3), (sink, 7)):
        torch.testing.assert_close(value.grad, torch.full_like(value, expected))
    if with_shared:
        torch.testing.assert_close(shared.grad, torch.full_like(shared, 5))
