# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The MXFP4 selector's forward and backward against the metadata the attention entry carries.

The launches are all this module does: the one QLI schedule (a layer that scores every visible
entry and the pool source, which are different schedules because the capacity sizes the candidate
writer), QSLI's schedule plus the per-row count it is bound by, and the BF16 SLIKG metadata its
teacher edge rides.  The fakes below stand in for the kernels and record what they were handed, so
these tests pin the wiring -- which schedule each branch launches, which options travel with it, and
that the backward reuses the same entry.
"""

from types import SimpleNamespace

import pytest
import torch
from torchao.quantization.qat import QATStep
from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_lightning_indexer import (
    DeepSeekV41QuantLightningIndexerConfig,
    _quant_v41_lightning_indexer_transform,
)
from torchao_npu.quantized_modules.deepseek_v4_1 import lightning_indexer

_TOKENS = 4
_POOL = 3
_BLOCK_SIZE = 2


def _indexer(*, pool=_POOL, block_size=_BLOCK_SIZE):
    """The indexer's AI_CPU metadata: the quantized pair, the teacher's backward, and the count."""
    return SimpleNamespace(
        fwd=torch.tensor([7], dtype=torch.int32),
        bwd=torch.tensor([5], dtype=torch.int32),
        fwd_sparse=torch.tensor([9], dtype=torch.int32),
        sparse_length=torch.tensor([[2], [1], [3], [1]], dtype=torch.int32),
    )


def _metadata(tokens=_TOKENS, compressed_tokens=2, *, pool=_POOL, block_size=_BLOCK_SIZE):
    """The attention metadata a layer is handed: frames in, metadata out."""
    return SimpleNamespace(
        q=SimpleNamespace(
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
            seqused=torch.tensor([tokens], dtype=torch.int32),
        ),
        cmp_k=SimpleNamespace(
            cu_seqlens=torch.tensor([0, compressed_tokens], dtype=torch.int32),
            residual=torch.zeros(1, dtype=torch.int32),
            seqused=torch.tensor([compressed_tokens], dtype=torch.int32),
        ),
        aicpu_metadata=SimpleNamespace(lightning_indexer=_indexer(pool=pool, block_size=block_size)),
    )


def _selector(*, mode="full", candidate_topk_blocks=_POOL):
    selector = lightning_indexer.QuantV41LightningIndexer()
    selector.index_topk = 4
    selector.compress_ratio = 2
    selector.candidate_topk_blocks = candidate_topk_blocks
    selector.candidate_block_size = _BLOCK_SIZE if candidate_topk_blocks > 0 else -1
    selector.mode = mode
    selector.has_candidate_pool = candidate_topk_blocks > 0
    selector.num_global_queries = torch.tensor(16.0)
    return selector


@pytest.fixture
def bypass_mxfp4_pack(monkeypatch):
    monkeypatch.setattr(lightning_indexer, "_pack_mxfp4", lambda value: (value, value.new_zeros(1)))


def _inputs():
    return (
        torch.zeros(1, _TOKENS, 2, 64, dtype=torch.bfloat16, requires_grad=True),
        torch.zeros(1, 2, 64, dtype=torch.bfloat16, requires_grad=True),
        torch.zeros(1, _TOKENS, 2, dtype=torch.float32, requires_grad=True),
    )


@pytest.mark.parametrize(
    "query_count, expected_teacher",
    [(8.0, 1.0), (32.0, 0.25), (None, None)],
    ids=["two-rows", "eight-rows", "missing-count"],
)
def test_qli_source_sorts_selection_publishes_pool_and_runs_slikg(
    monkeypatch, bypass_mxfp4_pack, query_count, expected_teacher
):
    q, k, w = _inputs()
    selector = _selector()
    selector.num_global_queries = None if query_count is None else torch.tensor(query_count)
    metadata = _metadata()
    metadata.q.seqused = torch.tensor([2], dtype=torch.int32)
    metadata.cmp_k.seqused = torch.tensor([1], dtype=torch.int32)
    unsorted = torch.tensor([3, -1, 1, 2], dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    candidate_blocks = torch.tensor(
        [[[0, 1, -1]], [[0, -1, -1]], [[1, 0, -1]], [[1, -1, -1]]],
        dtype=torch.int32,
    )
    seen = {}

    def qli(*args, **kwargs):
        seen["qli"] = (args, kwargs)
        return unsorted, torch.empty(0), candidate_blocks, torch.empty(0)

    def slikg(**kwargs):
        seen["slikg"] = kwargs
        return (
            torch.full_like(kwargs["q"], 2),
            torch.full_like(kwargs["k"], 3),
            torch.full_like(kwargs["w"], 4),
            torch.empty(0),
        )

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(quant_lightning_indexer=qli),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad",
        slikg,
        raising=False,
    )

    indices, scores, candidate = selector(q, k, w, metadata, candidates_BL1C=None)
    # A later microbatch must not change the denominator saved with this graph.
    selector.num_global_queries = torch.tensor(64.0)
    if query_count is None:
        with pytest.raises(RuntimeError, match=r"Set Selector\.num_global_queries"):
            scores.backward(torch.full_like(scores, 8.0))
        assert "slikg" not in seen
        return
    scores.backward(torch.full_like(scores, 8.0))

    indexer = metadata.aicpu_metadata.lightning_indexer
    assert torch.equal(indices[0, 0], torch.tensor([3, 2, 1, -1], dtype=torch.int32))
    assert scores.shape == indices.shape and scores.requires_grad
    assert torch.equal(candidate, candidate_blocks.unsqueeze(0))
    # The pool source launches the schedule that was sized by the capacity, with the same capacity:
    # a schedule and a launch that disagree about the pool cannot be reconciled downstream.
    assert seen["qli"][1]["metadata"] is indexer.fwd
    assert seen["qli"][1]["candidate_topk_blocks"] == _POOL
    assert seen["qli"][1]["candidate_block_size"] == _BLOCK_SIZE
    assert seen["qli"][1]["return_value"] is False
    # The backward reads the BF16 metadata off the same entry, so the teacher edge costs no launch.
    assert seen["slikg"]["metadata"] is indexer.bwd
    assert torch.equal(seen["slikg"]["sparse_indices"], indices.reshape(4, 1, 4))
    assert torch.equal(seen["slikg"]["attn_softmax_l1_norm"], torch.full((4, 1, 4), expected_teacher))
    for options in (seen["qli"][1], seen["slikg"]):
        torch.testing.assert_close(options["seqused_q"], torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(options["seqused_k"], torch.tensor([1], dtype=torch.int32))
    assert torch.equal(q.grad, torch.full_like(q, 2))
    assert torch.equal(k.grad, torch.full_like(k, 3))
    assert torch.equal(w.grad, torch.full_like(w, 4))


def test_qsli_searcher_searches_the_precomputed_pool_and_passes_it_through(monkeypatch, bypass_mxfp4_pack):
    q, k, w = _inputs()
    selector = _selector(mode="reindex", candidate_topk_blocks=4)
    metadata = _metadata(pool=4)
    candidate = torch.tensor(
        [[[[0, 2, -1, -1]], [[1, -1, -1, -1]], [[0, 1, 2, -1]], [[2, -1, -1, -1]]]],
        dtype=torch.int32,
    )
    topk_indices = torch.arange(4, dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    seen = {}

    def qsli(*args, **kwargs):
        seen["qsli"] = (args, kwargs)
        return topk_indices, torch.empty(0)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(quant_sparse_lightning_indexer=qsli),
        raising=False,
    )

    indices, scores, candidate_out = selector(q, k, w, metadata, candidates_BL1C=candidate)

    indexer = metadata.aicpu_metadata.lightning_indexer
    assert indices.shape == scores.shape == (1, 4, 4)
    torch.testing.assert_close(candidate_out, candidate, rtol=0, atol=0)
    # The search is bound by the count the port derived for these frames, and by the schedule that
    # count sized: neither is rebuilt here, and neither depends on QLI having run.
    torch.testing.assert_close(seen["qsli"][0][4], candidate.squeeze(0), rtol=0, atol=0)
    assert seen["qsli"][0][5] is indexer.sparse_length
    assert seen["qsli"][0][7:] == (1, _BLOCK_SIZE)
    assert seen["qsli"][1]["metadata"] is indexer.fwd_sparse
    assert seen["qsli"][1]["descale_k"].numel() == 1
    torch.testing.assert_close(seen["qsli"][1]["seqused_q"], torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(seen["qsli"][1]["seqused_k"], torch.tensor([2], dtype=torch.int32))


def test_qli_without_candidate_pool_runs_the_pool_free_schedule(monkeypatch, bypass_mxfp4_pack):
    q, k, w = _inputs()
    selector = _selector(candidate_topk_blocks=-1)
    metadata = _metadata(pool=-1, block_size=-1)
    candidate_blocks = torch.zeros((4, 1, 1), dtype=torch.int32)
    topk_indices = torch.arange(4, dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    seen = {}

    def qli(*args, **kwargs):
        seen["qli"] = (args, kwargs)
        return topk_indices, torch.empty(0), candidate_blocks, torch.ones_like(candidate_blocks)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(quant_lightning_indexer=qli),
        raising=False,
    )

    indices, scores, candidate = selector(q, k, w, metadata, candidates_BL1C=None)

    indexer = metadata.aicpu_metadata.lightning_indexer
    assert indices.shape == scores.shape == (1, 4, 4)
    assert candidate is None
    # A selector that scores every visible entry launches the pool-free schedule, and the kernels'
    # own "no pool" sentinel with it.
    assert seen["qli"][1]["metadata"] is indexer.fwd
    assert seen["qli"][1]["candidate_topk_blocks"] == -1
    assert seen["qli"][1]["candidate_block_size"] == -1


def test_a_pool_free_layer_inside_a_pool_geometry_scores_every_entry(monkeypatch, bypass_mxfp4_pack):
    """The released topology mixes both branches in one geometry.

    The ratio-1 source builds the pool that the metadata are sized by, while the ratio-2 layers
    score every visible entry -- they launch the pool-free schedule of the same geometry, which is
    a different layer's role, not a disagreement about the geometry.
    """
    q, k, w = _inputs()
    selector = _selector(candidate_topk_blocks=-1)
    metadata = _metadata(pool=_POOL, block_size=_BLOCK_SIZE)
    candidate_blocks = torch.zeros((4, 1, 1), dtype=torch.int32)
    topk_indices = torch.arange(4, dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    seen = {}

    def qli(*args, **kwargs):
        seen["qli"] = (args, kwargs)
        return topk_indices, torch.empty(0), candidate_blocks, torch.ones_like(candidate_blocks)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(quant_lightning_indexer=qli),
        raising=False,
    )

    indices, scores, candidate = selector(q, k, w, metadata, candidates_BL1C=None)

    indexer = metadata.aicpu_metadata.lightning_indexer
    assert indices.shape == scores.shape == (1, 4, 4)
    assert candidate is None
    assert seen["qli"][1]["metadata"] is indexer.fwd
    assert seen["qli"][1]["candidate_topk_blocks"] == -1
    assert seen["qli"][1]["candidate_block_size"] == -1


def test_a_candidate_table_disagreeing_with_the_layer_role_is_refused(bypass_mxfp4_pack):
    """Which kernel runs is decided by the table, so a role that lost it must not fall through.

    A source is handed nothing and publishes the pool; a searcher is handed the source's table.
    Without the check a searcher missing its table would quietly launch QLI and score every entry,
    which is the branch the pool exists to avoid.
    """
    q, k, w = _inputs()
    table = torch.zeros((1, 4, 1, _POOL), dtype=torch.int32)

    with pytest.raises(ValueError, match="cannot consume a candidate table"):
        _selector()(q, k, w, _metadata(), candidates_BL1C=table)
    with pytest.raises(ValueError, match="requires the source's candidate table"):
        _selector(mode="reindex", candidate_topk_blocks=_POOL)(q, k, w, _metadata(), candidates_BL1C=None)


def test_pack_mxfp4_uses_ds41_byte_layout(monkeypatch):
    q = torch.zeros(2, 3, 64, dtype=torch.bfloat16)
    calls = {}

    def dynamic_mx_quant(value, **kwargs):
        calls["value"] = value
        calls["kwargs"] = kwargs
        return torch.zeros(2, 3, 32, dtype=torch.uint8), torch.zeros(2, 3, 1, 2, dtype=torch.uint8)

    monkeypatch.setattr(lightning_indexer.torch_npu, "npu_dynamic_mx_quant", dynamic_mx_quant)

    data, scale = lightning_indexer._pack_mxfp4(q)

    assert data.shape == (2, 3, 32) and data.dtype is torch.uint8
    assert scale.shape == (2, 3, 1, 2) and scale.dtype is torch.uint8
    assert calls["value"].is_contiguous()
    assert calls["kwargs"]["block_size"] == 32


def test_module_swap_installs_and_restores_the_v41_selector_forward(monkeypatch):
    class SelectorHost(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return "original", args, kwargs

    host = SelectorHost()
    original_forward = host.forward
    captured = {}

    def quantized_forward(self, q, k, w, attention_metadata, *, candidates_BL1C):
        captured.update(self=self, args=(q, k, w, attention_metadata), candidate=candidates_BL1C)
        return "quantized"

    monkeypatch.setattr(lightning_indexer.QuantV41LightningIndexer, "forward", quantized_forward)
    _quant_v41_lightning_indexer_transform(host, DeepSeekV41QuantLightningIndexerConfig())
    operands = tuple(object() for _ in range(4))
    candidate = object()

    assert host(*operands, candidates_BL1C=candidate) == "quantized"
    assert captured == {"self": host, "args": operands, "candidate": candidate}

    _quant_v41_lightning_indexer_transform(host, DeepSeekV41QuantLightningIndexerConfig(step=QATStep.CONVERT))
    result = host(*operands, candidates_BL1C=candidate)

    assert result[0] == "original"
    assert host._torchao_npu_original_forward == original_forward
