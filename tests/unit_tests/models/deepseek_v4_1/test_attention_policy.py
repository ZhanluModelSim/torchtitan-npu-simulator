# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1 import (
    V41_CANDIDATE_SOURCE_LAYER,
    V41_FULL_COMPRESS_RATIOS,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
    model_registry,
)
from torchtitan_npu.models.deepseek_v4_1.attention import Attention
from torchtitan_npu.models.deepseek_v4_1.indexer import (
    FULL,
    REINDEX,
    REUSE,
    _selection_mask,
    _student_logits,
)


def test_index_selection_mask_is_document_isolated_and_causal() -> None:
    """Entry j is visible to query t of the same document iff group j is complete at t."""
    # Two documents of four tokens each; entry j covers tokens [2j, 2j + 2) of its own
    # document, so the whole-row pool holds two entries per document.
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 2, 4], dtype=torch.int32)
    visible, newest, newest_valid, pool_start = _selection_mask(
        cu_seqlens,
        cu_seqlens_k,
        num_tokens=8,
        num_cmp=4,
        compress_ratio=2,
    )

    expected_visible = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, True, False],
            [False, False, True, False],
            [False, False, True, True],
        ]
    )
    torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        newest.squeeze(-1),
        torch.tensor([-1, 0, 0, 1, 1, 2, 2, 3]),
        rtol=0,
        atol=0,
    )
    # A fresh document with no complete group yet pins no block.
    torch.testing.assert_close(
        newest_valid.squeeze(-1),
        torch.tensor([False, True, True, True, False, True, True, True]),
        rtol=0,
        atol=0,
    )
    # The reach is an interval of the whole-row pool, so the selection is numbered from the
    # query's own document: the second document's entries are 2 and 3 there, 0 and 1 here.
    torch.testing.assert_close(
        pool_start.squeeze(-1),
        torch.tensor([0, 0, 0, 0, 2, 2, 2, 2], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_student_logits_rescore_a_document_local_selection() -> None:
    """The rescore reads a selection in the numbering the selection op emitted.

    Two documents of four tokens, two entries each at ratio 2: document 1's local ids 0/1 name
    pool entries 2/3, and ``-1`` is the value softmax drops.
    """
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 2, 4], dtype=torch.int32)
    num_tokens, heads, dim, topk = 8, 2, 3, 2
    torch.manual_seed(0)
    idx_q = torch.randn(1, num_tokens, heads, dim, dtype=torch.float64)
    idx_k = torch.randn(1, cu_seqlens_k[-1].item(), dim, dtype=torch.float64)
    weights = torch.randn(1, num_tokens, heads, dtype=torch.float64)

    # Per row, ``(local id, the pool entry it names)``: padding in both slots and documents.
    picks = [
        [(0, 0), (-1, -1)],
        [(1, 1), (0, 0)],
        [(1, 1), (1, 1)],
        [(0, 0), (1, 1)],
        [(0, 2), (-1, -1)],
        [(1, 3), (0, 2)],
        [(1, 3), (1, 3)],
        [(0, 2), (1, 3)],
    ]
    local_BLK = torch.tensor([[[local for local, _ in row] for row in picks]])

    scores_BLK = _student_logits(idx_q, idx_k, weights, local_BLK, cu_seqlens, cu_seqlens_k)

    expected_BLK = torch.full((1, num_tokens, topk), -torch.inf, dtype=torch.float64)
    for token, row in enumerate(picks):
        for slot, (local, entry) in enumerate(row):
            if local < 0:
                continue
            dot = torch.einsum("hd,d->h", idx_q[0, token], idx_k[0, entry])
            expected_BLK[0, token, slot] = (dot.relu() * weights[0, token]).sum()

    torch.testing.assert_close(scores_BLK, expected_BLK, rtol=1e-12, atol=1e-12)
    # Marking agrees with the indices slot for slot: ``-1`` is the only un-scored slot.
    assert torch.equal(torch.isfinite(scores_BLK), local_BLK >= 0)
    # Row 4 is document 1's first token: reading its local 0 row-wide would score entry 0.
    row_wide = (torch.einsum("hd,d->h", idx_q[0, 4], idx_k[0, 0]).relu() * weights[0, 4]).sum()
    assert not torch.isclose(scores_BLK[0, 4, 0], row_wide)


def test_v41_config_uses_specialized_attention_with_explicit_ownership() -> None:
    config = model_registry("deepseek_v4_1_debugmodel").model

    assert not hasattr(config, "hc_head")
    assert config.kv_source_layers == V41_KV_SOURCE_LAYERS
    assert config.index_source_layers == V41_FULL_INDEX_SOURCE_LAYERS

    kv_sources = set(V41_KV_SOURCE_LAYERS)
    index_sources = set(V41_FULL_INDEX_SOURCE_LAYERS)
    key_owners = kv_sources & index_sources
    for layer_id, layer in enumerate(config.layers):
        assert isinstance(layer.attention, Attention.Config)
        # Every layer compresses and indexes; only a source carries the weights.
        assert layer.attention.compressor.is_source == (layer_id in kv_sources)
        assert (layer.attention.compressor.wkv is not None) == (layer_id in kv_sources)
        # The indexer's own mode spells out the same roles: it owns the index keys
        # (Full), only rescores them (Reindex), or computes nothing (Reuse).
        expected_mode = FULL if layer_id in key_owners else REINDEX if layer_id in index_sources else REUSE
        assert layer.attention.indexer.mode is expected_mode
        assert (layer.attention.indexer.wq_b is not None) == (layer_id in index_sources)
        assert (layer.attention.indexer.wk is not None) == (layer_id in key_owners)

    ratio_one_source = config.layers[20].attention
    assert ratio_one_source.compress_ratio == 1
    assert ratio_one_source.compressor.is_source
    assert ratio_one_source.compressor.wgate is None
    assert not hasattr(ratio_one_source.compressor, "use_ape")
    assert ratio_one_source.indexer.mode is FULL
    assert ratio_one_source.indexer.wk is not None
    assert ratio_one_source.indexer.k_norm is not None

    reindex = config.layers[24].attention
    assert not reindex.compressor.is_source
    # A reusing compressor holds no weights at all, so the checkpoint keys are the ones
    # the sources publish.
    assert reindex.compressor.wkv is None
    assert reindex.compressor.wgate is None
    assert reindex.compressor.norm is None
    assert reindex.compressor.rope is None
    assert reindex.indexer.mode is REINDEX
    # A re-indexing layer scores the shared keys, so it owns no key projection.
    assert reindex.indexer.wk is None
    assert reindex.indexer.k_norm is None
    assert not hasattr(reindex.indexer, "compressor")

    reuse = config.layers[21].attention
    assert reuse.indexer.mode is REUSE
    assert reuse.indexer.wq_b is None
    assert reuse.indexer.wk is None


def test_candidate_pool_roles_are_declared_per_indexer_layer() -> None:
    """The pool source and its consumers are config policy, not layer-id lookups.

    The capacity and the block size are read off the *selector*, which is the node that
    builds or searches the pool; the indexer only carries the candidate tensor between
    layers.  A layer outside the hierarchy spells the pool off with the kernels' negative
    sentinel rather than a local zero.
    """
    config = model_registry("deepseek_v4_1_debugmodel").model
    # The frozen debug recipe: one pool built at layer 20, consumed by every later
    # index source, over blocks of 8 slots.
    pool_consumers = {24, 28, 32, 36}

    for layer_id, layer in enumerate(config.layers):
        indexer = layer.attention.indexer
        selector = indexer.selector
        if layer_id == V41_CANDIDATE_SOURCE_LAYER:
            # The Full Mode source is the only layer that builds the pool.
            assert indexer.mode is FULL
            assert selector.candidate_topk_blocks == 4
            assert selector.candidate_block_size == 8
        elif layer_id in pool_consumers:
            # A Reindex Mode layer after the source searches the shared pool.
            assert indexer.mode is REINDEX
            assert selector.candidate_topk_blocks == 4
            assert selector.candidate_block_size == 8
        else:
            # Everything else scores every visible entry, or checks nothing at all.
            assert selector.candidate_topk_blocks == -1
            assert selector.candidate_block_size == -1

    assert config.layers[V41_CANDIDATE_SOURCE_LAYER].attention.indexer.mode is FULL
    assert config.layers[24].attention.indexer.mode is REINDEX


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"compress_ratios": V41_FULL_COMPRESS_RATIOS[:-1]}, "compress_ratios must match n_layers"),
        (
            {"compress_ratios": (0, 0, 2, 1) + (2,) * 16 + (1,) * 20},
            "consumes the compressed KV of layer 2",
        ),
        ({"kv_source_layers": (8, 14, 20)}, "compress_ratio=2 but no KV source precedes it"),
        (
            {"index_source_layers": tuple(layer for layer in V41_FULL_INDEX_SOURCE_LAYERS if layer != 20)},
            "scores the shared index keys of layer 14",
        ),
        ({"index_source_layers": (24, 28, 32, 36)}, "re-indexing layer 24 has no preceding index source"),
        (
            {"candidate_source_layer": 24, "index_source_layers": (2, 8, 14, 20, 36)},
            "candidate-pool source must also be an index source",
        ),
        ({"candidate_source_layer": 24}, "candidate-pool source must own the compressed KV"),
    ],
    ids=[
        "ratio_table_length",
        "kv_ratio_mismatch",
        "no_kv_source",
        "index_ratio_mismatch",
        "index_source_missing",
        "candidate_source_not_index",
        "candidate_source_owns_no_kv",
    ],
)
def test_invalid_reuse_topology_is_rejected(overrides, match, monkeypatch) -> None:
    """A consumer derives its plan from its own ratio, so a mismatched source fails at build."""
    import torchtitan_npu.models.deepseek_v4_1 as registry

    make_config = registry._make_v41_config

    def invalid_config(**kwargs):
        return make_config(**(kwargs | overrides))

    monkeypatch.setattr(registry, "_make_v41_config", invalid_config)
    with pytest.raises(ValueError, match=match):
        registry.model_registry("deepseek_v4_1_debugmodel_text")
