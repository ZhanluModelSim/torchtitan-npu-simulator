# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 cross-layer attention state is threaded, not stored.

The shared tensors (the compressed KV container, the index keys, the selected
container slots, the indexer's student logits and the candidate pool) travel
through the block and attention forward signatures.  These tests pin the
producer -> consumer chain and the loud failure of a consumer that is handed
nothing; the same path runs end to end in the 2-card integration case
``dsv41_debugmodel_2p_ep2_fsdp2``.
"""

import importlib

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import layer_metadata, preprocess
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1 import (
    V41_CANDIDATE_SOURCE_LAYER,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
)
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.indexer import REUSE

_TOKENS = torch.arange(128).remainder(32).unsqueeze(0)
_POSITIONS = torch.arange(128).unsqueeze(0)
# Block return slots: (x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates).
_CMP_K, _IDX_K, _TOPK, _TOPK_SCORES, _CANDIDATES = 2, 3, 4, 5, 6


def _tiny_debug_model(monkeypatch):
    """The registered debug flavor at CPU-sized widths, entered through the registry."""
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    make_config = registry._make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        )
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 64
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    return model, config


def _record_module_forwards(model):
    """Record each block's incoming/outgoing thread and the sparse core's inputs."""
    blocks_in, blocks_out, cores_in, handles = {}, {}, {}, []
    for name, layer in model.layers.items():

        def record_block(module, args, kwargs, output, *, name=name):
            blocks_in[name] = dict(kwargs)
            blocks_out[name] = output

        handles.append(layer.register_forward_hook(record_block, with_kwargs=True))

        def record_core(module, args, kwargs, output, *, name=name):
            # The core's signature is (q, swa_k, attn_sink, attention_metadata, *, cmp_k, ...),
            # so the compressed container travels as a keyword like the selection does.
            cores_in[name] = (kwargs.get("cmp_k"), kwargs.get("topk_indices"))

        handles.append(layer.attention.inner_attention.register_forward_hook(record_core, with_kwargs=True))
    return blocks_in, blocks_out, cores_in, handles


def test_cross_layer_state_is_threaded_from_source_to_consumer(monkeypatch) -> None:
    """Each consumer receives the object its source published, not a copy."""
    model, _ = _tiny_debug_model(monkeypatch)
    blocks_in, blocks_out, cores_in, handles = _record_module_forwards(model)
    try:
        _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
        model(_TOKENS, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    # Window-only layers own no compressed KV and are handed none.
    for name in ("0", "1"):
        assert blocks_in[name]["cmp_k"] is None
        assert blocks_out[name][_CMP_K] is None

    # Layer 2 is the first KV/index source: it publishes, its group consumes the same objects.
    first_container = blocks_out["2"][_CMP_K]
    first_index_key = blocks_out["2"][_IDX_K]
    first_selection = blocks_out["2"][_TOPK]
    assert blocks_in["2"]["cmp_k"] is None
    assert first_container is not None and first_index_key is not None and first_selection is not None
    for name in ("3", "4", "5", "6", "7"):
        assert blocks_in[name]["cmp_k"] is first_container
        assert blocks_in[name]["idx_k"] is first_index_key
        assert blocks_in[name]["topk_indices"] is first_selection
        assert blocks_out[name][_CMP_K] is first_container

    # A source supersedes what is in flight: each group's source receives the previous
    # group's container and publishes its own, which its own group then consumes.
    for source, group in (("8", ("9", "10", "11", "12", "13")), ("14", ("15", "16", "17", "18", "19"))):
        assert blocks_in[source]["cmp_k"] is not None
        group_container = blocks_out[source][_CMP_K]
        assert group_container is not first_container
        for name in group:
            assert blocks_in[name]["cmp_k"] is group_container
            assert blocks_out[name][_CMP_K] is group_container

    # The ratio-1 group shares the layer-20 global-KV container through to the last layer.
    container = blocks_out["20"][_CMP_K]
    assert blocks_in["20"]["cmp_k"] is not container
    for name in (str(layer_id) for layer_id in range(21, 40)):
        assert blocks_in[name]["cmp_k"] is container
        assert blocks_out[name][_CMP_K] is container

    # A re-indexing layer returns the shared keys unchanged and publishes a fresh selection.
    shared_index_key = blocks_out["20"][_IDX_K]
    shared_selection = blocks_out["20"][_TOPK]
    for name in ("21", "22", "23"):
        assert blocks_in[name]["idx_k"] is shared_index_key
        assert blocks_in[name]["topk_indices"] is shared_selection
    assert blocks_in["24"]["idx_k"] is shared_index_key
    assert blocks_out["24"][_IDX_K] is shared_index_key
    reselected = blocks_out["24"][_TOPK]
    assert reselected is not shared_selection
    for name in ("25", "26", "27"):
        assert blocks_in[name]["idx_k"] is shared_index_key
        assert blocks_in[name]["topk_indices"] is reselected

    # The candidate pool is built once, at layer 20, and reaches the layers inside its window.
    pool = blocks_out["20"][_CANDIDATES]
    assert pool is not None
    for name in ("21", "22", "23"):
        assert blocks_out[name][_CANDIDATES] is pool
    for name in ("24", "28", "32", "36", "39"):
        assert blocks_in[name]["candidates"] is pool
        assert blocks_out[name][_CANDIDATES] is pool

    # The distillation loss attaches to every layer that consumes a selection, so the
    # index source publishes the student logits and its group scores the same tensor
    # against its own attention mass; the window-only layers own none.
    for name in ("0", "1"):
        assert blocks_in[name]["topk_scores"] is None
        assert blocks_out[name][_TOPK_SCORES] is None
    source_scores = blocks_out["2"][_TOPK_SCORES]
    assert source_scores is not None
    for name in ("3", "4", "5", "6", "7"):
        assert blocks_in[name]["topk_scores"] is source_scores
        assert blocks_out[name][_TOPK_SCORES] is source_scores
    shared_scores = blocks_out["20"][_TOPK_SCORES]
    assert shared_scores is not None
    assert blocks_in["24"]["topk_scores"] is shared_scores

    # The sparse core receives the threaded objects themselves.
    core_container, core_selection = cores_in["0"]
    assert core_container is None and core_selection is None
    core_container, core_selection = cores_in["4"]
    assert core_container is first_container
    assert core_selection is first_selection
    core_container, core_selection = cores_in["21"]
    assert core_container is container
    assert core_selection is shared_selection
    core_container, core_selection = cores_in["30"]
    assert core_container is container
    assert core_selection is blocks_out["28"][_TOPK]


def test_candidate_pool_reaches_only_its_window_indexers(monkeypatch) -> None:
    """The pool built at layer 20 is the mask the later indexers actually select with."""
    model, _ = _tiny_debug_model(monkeypatch)
    indexer_module = importlib.import_module("torchtitan_npu.models.deepseek_v4_1.indexer")
    registered_forward = indexer_module.HierarchicalIndexer.forward
    observed_masks = []

    def record_forward(self, x, qr, positions, attention_metadata, **kwargs):
        observed_masks.append(kwargs.get("candidates"))
        return registered_forward(self, x, qr, positions, attention_metadata, **kwargs)

    monkeypatch.setattr(indexer_module.HierarchicalIndexer, "forward", record_forward)
    _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
    with torch.no_grad():
        model(_TOKENS, **kwargs)

    # One call per layer, in layer order: the pool is built at layer 20 and reaches
    # every later layer, whatever role that layer plays.
    assert len(observed_masks) == len(model.layers)
    pool_source = V41_CANDIDATE_SOURCE_LAYER
    assert all(mask is None for mask in observed_masks[: pool_source + 1])
    pool = observed_masks[pool_source + 1]
    assert pool is not None
    assert all(mask is pool for mask in observed_masks[pool_source + 1 :])


def test_metadata_document_boundaries_restart_per_packed_document(monkeypatch) -> None:
    """The varlen metadata is the row's boundaries, derived from the positions, and its pool grid.

    One entry per ``(backend, compress_ratio)`` -- the entry *is* one geometry -- and the row is
    all a *reference* backend reads: ``cu_seq_k`` is that backend's own ratio applied to those
    boundaries once, where the ratio is known, so a layer that pools reads a boundary set instead
    of dividing one it was handed.  The fused port's frames and the kernel records on top of that
    row are the override tests' subject.
    """
    model, _ = _tiny_debug_model(monkeypatch)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]], dtype=torch.long)
    tokens = torch.zeros_like(positions)

    _, _, extra = preprocess(model, tokens, tokens, positions)
    metadata = extra["attention_metadata"]
    assert len(metadata) == 3, "one entry per (backend, compress_ratio) geometry"
    # Three documents, so the boundaries are the row start plus every reset after it, and the
    # query's own positions are the ones the row carries.  Every geometry shares the row and
    # differs only in the key grid it folded there.
    expected_cu = torch.tensor([0, 4, 8, 10], dtype=torch.int32)
    for (backend, ratio), row in metadata.items():
        assert backend is CompressedSparseInnerAttention2.Config
        torch.testing.assert_close(row.cu_seq_q, expected_cu, rtol=0, atol=0)
        expected_pool = expected_cu if ratio <= 1 else (expected_cu // ratio).to(torch.int32)
        torch.testing.assert_close(row.cu_seq_k, expected_pool, rtol=0, atol=0)
        torch.testing.assert_close(row.positions_q, positions)


def test_weightless_compressor_passes_the_container_through(monkeypatch) -> None:
    """Every layer compresses; a non-source one holds no weights and returns its input."""
    model, config = _tiny_debug_model(monkeypatch)
    kv_sources = set(V41_KV_SOURCE_LAYERS)
    for layer_id, layer in model.layers.items():
        compressor = layer.attention.compressor
        assert compressor.is_source == (int(layer_id) in kv_sources)
        if int(layer_id) not in kv_sources:
            assert not list(compressor.parameters())

    # A window-only layer owns nothing and is handed nothing.
    returned, latent = model.layers["0"].attention.compressor(torch.zeros(1, 128, config.dim), _POSITIONS)
    assert returned is None
    assert latent is None

    # A reuse layer hands back the very container it was given, with no latent.
    handed = torch.zeros(1, 64, config.layers[3].attention.head_dim)
    returned, latent = model.layers["3"].attention.compressor(
        torch.zeros(1, 128, config.dim),
        _POSITIONS,
        handed,
    )
    assert returned is handed
    assert latent is None


def test_student_logits_are_published_outside_training(monkeypatch) -> None:
    """The indexer always builds the student logits; only the loss decides to consume them.

    The rescore at the selected entries is unconditional, so an eval forward under
    ``torch.no_grad()`` returns the same contract as a training one: a window-only layer
    publishes ``None`` and an index source publishes its logits, which its whole group
    shares.
    """
    model, _ = _tiny_debug_model(monkeypatch)
    model.eval()
    blocks_in, blocks_out, _, handles = _record_module_forwards(model)
    try:
        _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
        with torch.no_grad():
            model(_TOKENS, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    for name in ("0", "1"):
        assert blocks_in[name]["topk_scores"] is None
        assert blocks_out[name][_TOPK_SCORES] is None
    source_scores = blocks_out["2"][_TOPK_SCORES]
    assert source_scores is not None
    # A real rescore, not an empty placeholder: at least the reachable slots are scored.
    assert torch.isfinite(source_scores).any()
    for name in ("3", "4", "5", "6", "7"):
        assert blocks_in[name]["topk_scores"] is source_scores
        assert blocks_out[name][_TOPK_SCORES] is source_scores


def test_unreachable_selection_slots_are_marked_in_the_student_logits(monkeypatch) -> None:
    """The indexer marks a padded selection slot with -inf, which is what the loss drops.

    The distillation reads row validity from ``isfinite``, so the marking and the indices
    have to agree on every slot of every layer's student logits.
    """
    model, _ = _tiny_debug_model(monkeypatch)
    _, blocks_out, _, handles = _record_module_forwards(model)
    try:
        _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
        model(_TOKENS, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    unreachable = 0
    for name, output in blocks_out.items():
        scores = output[_TOPK_SCORES]
        if scores is None:
            continue
        indices = output[_TOPK]
        assert torch.equal(torch.isfinite(scores), indices >= 0), name
        unreachable += int((indices < 0).sum())
    # Early queries cannot fill index_topk slots, so the marking is actually exercised.
    assert unreachable > 0


def test_weightless_indexer_passes_the_selection_through(monkeypatch) -> None:
    """Every layer indexes; a reuse layer holds no weights and returns its inputs."""
    model, config = _tiny_debug_model(monkeypatch)
    index_sources = set(V41_FULL_INDEX_SOURCE_LAYERS)
    for layer_id, layer in model.layers.items():
        indexer = layer.attention.indexer
        # Full Mode owns the keys, Reindex Mode only rescores them, and every other layer
        # is Reuse Mode.
        assert (indexer.mode is REUSE) == (int(layer_id) not in index_sources)
        if int(layer_id) not in index_sources:
            assert not list(indexer.parameters())

    hidden = torch.zeros(1, 128, config.dim)
    qr = torch.zeros(1, 128, config.layers[0].attention.q_lora_rank)

    # A window-only layer owns nothing and is handed nothing.
    assert model.layers["0"].attention.indexer(hidden, qr, _POSITIONS, None) == (None, None, None, None)

    # A reuse layer hands back the very tensors it was given.
    shared_key = torch.zeros(1, 64, config.layers[4].attention.indexer.index_head_dim)
    shared_selection = torch.zeros(1, 128, 4, dtype=torch.long)
    returned = model.layers["4"].attention.indexer(
        hidden,
        qr,
        _POSITIONS,
        None,
        idx_k=shared_key,
        topk_indices=shared_selection,
    )
    assert returned[0] is shared_key
    assert returned[1] is shared_selection
    assert returned[2] is None
    assert returned[3] is None


def test_reuse_layer_without_the_shared_tensor_fails_loudly(monkeypatch) -> None:
    """A consumption contract that is not met is an error, not a silent empty reuse."""
    model, config = _tiny_debug_model(monkeypatch)
    _, _, kwargs = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
    attention_metadata = layer_metadata(model.layers["3"], kwargs)
    hidden = torch.zeros(1, _TOKENS.shape[1], config.dim)

    # A layer that reuses the compressed KV must be handed one.
    with pytest.raises(AssertionError, match="reuses the compressed KV"):
        model.layers["3"].attention(hidden, _POSITIONS, attention_metadata)

    # A re-indexing layer must be handed the shared index keys; its compressed KV is
    # already in flight, so only the key is missing.
    head_dim = config.layers[24].attention.head_dim
    with pytest.raises(AssertionError, match="Reindex Mode indexer rescores"):
        model.layers["24"].attention(
            hidden,
            _POSITIONS,
            attention_metadata,
            cmp_k=torch.zeros(1, _TOKENS.shape[1], head_dim),
        )
