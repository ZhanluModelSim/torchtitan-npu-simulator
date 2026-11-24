# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4 SMLAG carries the raw teacher to the selector's SLIKG backward.

The sparse core no longer trains the indexer itself.  SMLAG's by-product -- the marginal
mass ``p`` over the selected slots -- is written on ``topk_scores``' gradient, with every
``-1`` slot zeroed. These cases pin that edge,
including the layers that must not emit one.
"""

import pytest
import torch

from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

pytestmark = pytest.mark.usefixtures("dsv4")


def _config(*, ratio=4):
    return ascendc.AscCompressedSparseInnerAttention.Config(
        window_size=2,
        compress_ratio=ratio,
        softmax_scale=0.5,
        index_topk=3,
        num_heads=1,
        head_dim=4,
        index_num_heads=1,
        index_head_dim=4,
    )


def _attention(config):
    return config.build()


def _metadata(config, positions):
    return config.build_attention_metadata(positions)


def _patch_smla(monkeypatch, *, junk=7.0):
    def forward(q, **kwargs):
        return q.clone(), torch.zeros(1, q.shape[0], q.shape[1])

    def backward(q, *args, **kwargs):
        indices = kwargs["cmp_sparse_indices"]
        if indices is None:
            # An HCA layer selects its whole visible prefix itself; SMLAG reports no
            # per-slot marginal there and the core must not invent one.
            p = None
        else:
            p = torch.zeros(indices.shape, dtype=torch.float32)
            valid = indices >= 0
            # The real op writes a non-zero marginal on unused slots too; the port must
            # clean them or SLIKG's ``p_reduce`` (Z) is polluted.
            p[..., 0] = torch.where(valid[..., 0], 0.1, junk)
            p[..., 1] = torch.where(valid[..., 1], 0.3, junk)
            p[..., 2] = torch.where(valid[..., 2], 0.4, junk)
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            p,
        )

    # The kernels are resolved at call time, so replacing the ops on the namespace is what
    # reaches the core this test builds.
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", forward)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)


def test_the_teacher_reaches_the_carrier_with_padding_zeroed(monkeypatch):
    _patch_smla(monkeypatch)
    config = _config()
    attention = _attention(config)
    metadata = _metadata(config, torch.arange(4))
    indices = torch.tensor([[[0, 1, -1], [0, -1, -1], [-1, -1, -1], [0, 1, 2]]], dtype=torch.int64).squeeze(0)
    scores = torch.zeros(4, 3, requires_grad=True)
    q = torch.zeros(4, 1, 4, dtype=torch.bfloat16)

    out = attention(
        q,
        q.squeeze(1),
        torch.zeros(1),
        cmp_k=torch.zeros(1, 4, dtype=torch.bfloat16),
        topk_indices=indices,
        topk_scores=scores,
        attention_metadata=metadata,
    )
    out.sum().backward()

    assert scores.grad is not None
    expected = torch.zeros(4, 3)
    expected[0] = torch.tensor([0.1, 0.3, 0.0])
    expected[1] = torch.tensor([0.1, 0.0, 0.0])
    expected[3] = torch.tensor([0.1, 0.3, 0.4])
    torch.testing.assert_close(scores.grad, expected, rtol=0, atol=0)
    assert (scores.grad[indices < 0] == 0).all()


@pytest.mark.parametrize("mode", ["eval", "hca"])
def test_layers_that_must_not_emit_a_teacher(monkeypatch, mode):
    _patch_smla(monkeypatch)
    ratio = 128 if mode == "hca" else 4
    config = _config(ratio=ratio)
    attention = _attention(config)
    attention.train(mode != "eval")
    tokens = 128 if ratio == 128 else 4
    metadata = _metadata(config, torch.arange(tokens))
    q = torch.zeros(tokens, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    # A window-only/HCA layer has a compressed pool but no selector, so no indices and no
    # carrier.  There is no teacher edge to feed.
    indices = torch.zeros(tokens, 3, dtype=torch.int64) if ratio == 4 else None
    scores = torch.zeros(tokens, 3, requires_grad=True) if ratio == 4 else None
    cmp_k = torch.zeros(tokens // ratio, 4, dtype=torch.bfloat16)

    out = attention(
        q,
        q.squeeze(1),
        torch.zeros(1),
        cmp_k=cmp_k,
        topk_indices=indices,
        topk_scores=scores,
        attention_metadata=metadata,
    )
    out.sum().backward()

    if scores is not None:
        # Eval has no indexer gradient to feed, so the carrier is not threaded through and
        # SMLAG never emits a teacher on it.
        assert scores.grad is None
