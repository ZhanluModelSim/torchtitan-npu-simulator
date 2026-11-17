# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The precomputed SMLA metadata: one metadata set per geometry, built by the port, read per layer.

``AscV41SparseAttention.Config`` -- the config that owns the AscendC attention -- builds its own
geometry's AI_CPU metadata operators' outputs at the metadata seam, off the per-layer hot path
(the forward, the full-AC recompute and the backward alike).  The entry *is* one geometry: the
port keys itself by ``(class, compress_ratio)``, so the map carries one metadata set per ratio the
stack runs, no builder has to know the stack's ratio table, and every layer reads a grid built
for its own kernels.  These tests pin the contracts that make the reuse safe: what one geometry
precomputes, that the map builds each geometry exactly once, that the metadata is the only source
of the kernels' metadata (a row that never went through the port fails instead of falling back to
an inline operator call), the pytree shape the compiled path flattens, and the CP half rebuilding
the metadata from this rank's own frames.
"""

from itertools import pairwise

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchtitan.config import derive

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import (
    cp_parallel_dims,
    install_plain_cp_shard,
    preprocess,
    tiny_v41_model,
)
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.common.metadata_extension import register_pytree_node_for_dataclass
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata
from torchtitan_npu.override.deepseek_v4_1.metadata import (
    AicpuGeometry,
    AicpuMetadata,
    KernelFrame,
)
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import (
    AscV41Metadata,
    AscV41SparseAttention,
    shard_asc_v41_metadata,
)
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set

_TOKENS = torch.arange(128).remainder(32).unsqueeze(0)
_POSITIONS = torch.arange(128).unsqueeze(0)

# The geometry the fused recipes state, at the widths the tiny debug flavor runs.  Stated by
# the override's arguments, not by the model: the port is the only thing that sizes metadata.
_TINY_GEOMETRY = dict(window_size=128, num_heads=2, head_dim=8, index_topk=32, index_num_heads=2, index_head_dim=4)
_TINY_AICPU_GEOMETRY = AicpuGeometry(**_TINY_GEOMETRY)


def _fake_metadata_ops(monkeypatch):
    """Count the AI_CPU metadata launches and keep each call's output and geometry.

    All four operators of one geometry: the SMLA forward/backward pair the attention reads and
    the indexer pair the fused selector reads, which ride the same entry.  ``raising=False``
    keeps the fakes installable on runners without ``cann_ops_transformer`` registered at all;
    the operators are never really run either way.
    """
    state = {"smla_fwd": [], "smla_bwd": [], "li": [], "slig": []}

    def _metadata_op(kind):
        def _call(*args, **kwargs):
            out = torch.zeros(1, dtype=torch.int32)
            state[kind].append((kwargs, out))
            return out

        return _call

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _metadata_op("smla_fwd"), raising=False)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _metadata_op("smla_bwd"), raising=False
    )
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer_metadata", _metadata_op("li"), raising=False)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad_metadata",
        _metadata_op("slig"),
        raising=False,
    )
    return state


def _core_config(
    *,
    compress_ratio: int = 1,
    window_size: int = 2,
    num_heads: int = 1,
    head_dim: int = 4,
    index_topk: int = 3,
    index_num_heads: int = 2,
    index_head_dim: int = 4,
):
    """An AscendC attention config: the reference core's fields plus the geometry its metadata needs."""
    return AscV41SparseAttention.Config(
        window_size=window_size,
        softmax_scale=0.5,
        compress_ratio=compress_ratio,
        num_heads=num_heads,
        head_dim=head_dim,
        index_topk=index_topk,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )


def _positions(bounds: list[int]) -> torch.Tensor:
    """One packed row of these documents, positions resetting at every start."""
    return torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)


def _reference_row(bounds: list[int], *, ratio: int) -> DeepSeekV41Metadata:
    """The row a fused entry starts from, built by the reference backend's own config."""
    config = CompressedSparseInnerAttention2.Config(window_size=2, softmax_scale=0.5, compress_ratio=ratio)
    return config.build_attention_metadata(_positions(bounds))


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_a_geometry_builds_its_own_ratios_frames_and_metadata(monkeypatch, ratio):
    """One config, one geometry: its own ratio's axis and exactly that geometry's metadata.

    The port's builder starts from the same row the reference core builds -- the backend that
    knows the ratio folds ``cu_seq_k`` there, once -- and adds only what a fused kernel reads:
    this geometry's frames and the metadata sized for it.  A ratio-0 layer has no compressed
    axis at all, and its single metadata is the window-only one; nothing here consults a
    stack-wide ratio table, because there is none.
    """
    state = _fake_metadata_ops(monkeypatch)
    config = _core_config(compress_ratio=ratio, num_heads=2, head_dim=4, index_topk=3)
    row = _positions([0, 4, 8])

    built = config.build_attention_metadata(row)

    assert isinstance(built, AscV41Metadata)
    # The entry names its own geometry, and that key is what the layer reads back.
    assert built.ratio == ratio
    assert config.attention_metadata_key == (AscV41SparseAttention.Config, ratio)
    # The row's own boundaries, and this ratio's pool grid folded by the backend that built them.
    assert built.cu_seq_q.tolist() == [0, 4, 8]
    assert built.cu_seq_k.tolist() == [bound // max(ratio, 1) for bound in (0, 4, 8)]
    assert built.q.cu_seqlens.tolist() == [0, 4, 8]
    assert built.swa_k.cu_seqlens.tolist() == [0, 4, 8]
    if ratio == 0:
        assert built.cmp_k is None, "a window-only layer has no compressed axis to grid"
    else:
        assert built.cmp_k is not None
        assert built.cmp_k.cu_seqlens.tolist() == [bound // ratio for bound in (0, 4, 8)]
        assert built.cmp_k.seqused.tolist() == [4 // ratio, 4 // ratio]
        if ratio == 1:
            # The uncompressed case: the compressed axis is the row itself, and the kernels
            # reject a residual for an axis they do not compress.
            assert built.cmp_k.residual is None
        else:
            assert built.cmp_k.residual is not None
            assert built.cmp_k.residual.tolist() == [0, 0]
    # One metadata set per operator pair -- and one launch each, not one per layer, which is the
    # point of building them here.
    assert len(state["smla_fwd"]) == 1
    assert len(state["smla_bwd"]) == 1
    metadata = built.aicpu_metadata
    assert isinstance(metadata, AicpuMetadata)
    assert metadata.sparse_attn.fwd is state["smla_fwd"][0][1]
    assert metadata.sparse_attn.bwd is state["smla_bwd"][0][1]
    # The indexer's metadata ride the same entry, and only where the layer owns a pool.
    assert len(state["li"]) == (1 if ratio > 0 else 0)
    assert len(state["slig"]) == (1 if ratio > 0 else 0)
    if ratio > 0:
        assert metadata.lightning_indexer.fwd is state["li"][0][1]
        assert metadata.lightning_indexer.bwd is state["slig"][0][1]
    else:
        assert metadata.lightning_indexer.fwd is None
        assert metadata.lightning_indexer.bwd is None
    # The geometry the metadata were sized with: this ratio, this selection width, this pool.
    assert metadata.geometry == AicpuGeometry(
        window_size=2, num_heads=2, head_dim=4, index_topk=3, index_num_heads=2, index_head_dim=4
    )
    selection = 3 if ratio > 0 else 0
    kwargs, _ = state["smla_fwd"][0]
    assert (kwargs["cmp_topk"], kwargs["has_cmp_kv"], kwargs["cmp_ratio"]) == (selection, ratio > 0, max(ratio, 1))
    # ... and the selector's metadata were sized with the config's own indexer numbers.
    if ratio > 0:
        indexer_kwargs, _ = state["li"][0]
        assert (indexer_kwargs["num_heads_q"], indexer_kwargs["head_dim"], indexer_kwargs["topk"]) == (2, 4, 3)


def test_the_reference_row_carries_no_metadata_and_the_port_always_builds(monkeypatch):
    """Two builders, two products: the row contract, and the row plus this geometry's metadata.

    The reference core reads the row's boundaries and its own ratio, so its builder produces
    that and nothing else -- no kernel geometry anywhere on its config.  The port's builder
    always adds metadata: its config states all five numbers, so an entry is always sized for
    *some* geometry, and the geometry it was sized with rides the entry.
    """
    reference = CompressedSparseInnerAttention2.Config(window_size=2, softmax_scale=0.5, compress_ratio=2)
    # The reference builder is geometry-free -- that its config cannot even state one is
    # ``test_fused_geometry``'s pin -- so one row entry serves the whole stack.
    row = reference.build_attention_metadata(_positions([0, 4]))
    assert type(row) is DeepSeekV41Metadata
    assert row.cu_seq_q.tolist() == [0, 4]
    assert row.cu_seq_k.tolist() == [0, 2]
    # No frames and no metadata: a row is not a fused entry, and nothing pretends otherwise.
    assert not hasattr(row, "q")
    assert not hasattr(row, "aicpu_metadata")

    state = _fake_metadata_ops(monkeypatch)
    port = _core_config(compress_ratio=2, num_heads=2, head_dim=4, index_topk=3)

    built = port.build_attention_metadata(_positions([0, 4]))

    # The numbers the metadata were sized with are the port's own, stated once on its config.
    assert built.aicpu_metadata.geometry == port.geometry
    assert built.cmp_k is not None
    assert len(state["smla_fwd"]) == len(state["smla_bwd"]) == len(state["li"]) == len(state["slig"]) == 1


def test_the_map_carries_one_entry_per_ratio_and_builds_each_once(monkeypatch):
    """One entry per geometry, named by the port and its ratio, built once per microbatch.

    The fused recipe swaps the core on **every** layer (``asc`` targets the core config with
    ``exact=True``), and the debug flavor runs ratios ``{0, 1, 2}``: the map therefore carries
    three entries -- one per geometry -- so the metadata operators run once per geometry per
    microbatch rather than once per layer, and each layer's own dispatch key finds the entry
    built for its own kernels.
    """
    state = _fake_metadata_ops(monkeypatch)
    model, config = tiny_v41_model(monkeypatch, configure=_fuse_every_layer)
    _, _, extra = preprocess(model, _TOKENS, _TOKENS, _POSITIONS)
    metadata = extra["attention_metadata"]

    assert len(config.layers) == 40
    assert all(type(layer.attention.inner_attention) is AscV41SparseAttention.Config for layer in config.layers)
    assert set(metadata) == {
        (AscV41SparseAttention.Config, 0),
        (AscV41SparseAttention.Config, 1),
        (AscV41SparseAttention.Config, 2),
    }
    # Every layer reads its own geometry's entry: the key carries the ratio, so no layer can be
    # handed another geometry's grid.
    for layer in model.layers.values():
        entry = metadata[layer.attention.attention_metadata_key]
        assert isinstance(entry, AscV41Metadata)
        assert entry.ratio == layer.attention.inner_attention.compress_ratio
        if entry.ratio == 0:
            assert entry.cmp_k is None
        else:
            assert entry.cmp_k is not None
    # Three geometries, so exactly three launches of each operator for the whole microbatch --
    # the window-only geometry included, whose single metadata is keyed 0.
    assert len(state["smla_fwd"]) == 3
    assert len(state["smla_bwd"]) == 3
    assert sum(1 for kwargs, _ in state["smla_fwd"] if not kwargs["has_cmp_kv"]) == 1
    # Two of them pool, so two indexer metadata sets.
    assert len(state["li"]) == 2
    assert len(state["slig"]) == 2
    # Every metadata was built for the geometry the override's arguments stated -- one numbers
    # source, the port's own config, rather than a per-layer restatement.
    for entry in metadata.values():
        assert entry.aicpu_metadata.geometry == _TINY_AICPU_GEOMETRY


def test_the_attention_reads_the_metadata_and_never_launches(monkeypatch):
    """The builder launches each operator once; every consumer after that launches nothing.

    There is no per-layer operator path left, so a forward, a recompute and a backward add
    exactly zero launches.
    """
    state = _fake_metadata_ops(monkeypatch)
    # The main operators: shape-preserving fakes, so this test can drive the core without
    # touching what it is counting -- the metadata operators above.
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **k: (q.clone(), torch.zeros(1, q.shape[0], q.shape[1])),
        raising=False,
    )

    def _grad(q, grad_output, output, lse, **kwargs):
        indices = kwargs["cmp_sparse_indices"]
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            torch.zeros(indices.shape, dtype=torch.float32),
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", _grad, raising=False)

    config = _core_config(compress_ratio=1, num_heads=1, head_dim=4, index_topk=3, index_num_heads=1, index_head_dim=4)
    attention = build_cpu_model(config)
    entry = config.build_attention_metadata(_positions([0, 4]))
    # The whole per-microbatch cost, paid once here: one launch of each operator, for this
    # geometry, before any layer runs.
    built = {kind: len(state[kind]) for kind in state}
    assert built == {"smla_fwd": 1, "smla_bwd": 1, "li": 1, "slig": 1}

    def consume(attention_metadata):
        q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
        return attention(
            q,
            q.squeeze(2),
            torch.zeros(1),
            attention_metadata,
            cmp_k=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
            topk_indices=torch.zeros(1, 4, 1, 3),
            topk_scores=torch.zeros(1, 4, 3, requires_grad=True),
        )

    # Two forward-and-backward cycles: four kernel passes over the same entry, and the counts do
    # not move -- the forward and the backward both read the metadata the builder produced.
    for _ in range(2):
        out = consume(entry)
        assert out.shape == (1, 4, 1, 4)
        out.sum().backward()
    assert {kind: len(state[kind]) for kind in state} == built


def test_a_row_that_never_went_through_the_port_is_refused(monkeypatch):
    """The metadata are the only source of the kernels' metadata: there is no inline fallback.

    The AscendC attention reads this geometry's frames and metadata off the entry it is handed.  A
    caller holding the reference row -- what the reference backend builds, with no geometry on
    it -- is holding the wrong metadata for the AscendC attention, and the forward fails on the missing
    field rather than paying for an AI_CPU call on the side.
    """
    state = _fake_metadata_ops(monkeypatch)
    attention = build_cpu_model(
        _core_config(compress_ratio=2, num_heads=1, head_dim=4, index_topk=3, index_num_heads=1, index_head_dim=4)
    )
    row = _reference_row([0, 4], ratio=2)
    q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)

    with pytest.raises(AttributeError, match="'DeepSeekV41Metadata' object has no attribute 'q'"):
        attention(
            q,
            q.squeeze(2),
            torch.zeros(1),
            row,
            cmp_k=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            topk_indices=torch.zeros(1, 4, 1, 3),
            topk_scores=torch.zeros(1, 4, 3),
        )
    # Refused, not served: nothing was launched for it.
    assert {kind: len(state[kind]) for kind in state} == {"smla_fwd": 0, "smla_bwd": 0, "li": 0, "slig": 0}


def test_the_shared_metadata_survive_every_consumer(monkeypatch):
    """One metadata set serves every layer, the recompute and the backward: nothing writes it.

    The kernels take ``metadata`` as a read-only input -- the AI_CPU operator's own output,
    consulted for the grid -- so sharing one metadata set across all consumers is sound; this pins
    this side of the premise, the operator side being validated by the end-to-end A5 runs the
    PR documents.  Through several forward and backward cycles every kernel call reads the
    very same tensor objects, and they stay bit-for-bit what the builder produced.
    """
    _fake_metadata_ops(monkeypatch)
    received_fwd, received_bwd = [], []

    def _mla(q, **kwargs):
        received_fwd.append(kwargs["metadata"])
        return q.clone(), torch.zeros(1, q.shape[0], q.shape[1])

    def _grad(q, grad_output, output, lse, **kwargs):
        received_bwd.append(kwargs["metadata"])
        indices = kwargs["cmp_sparse_indices"]
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            torch.zeros(indices.shape, dtype=torch.float32),
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", _mla, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", _grad, raising=False)

    config = _core_config(compress_ratio=1, num_heads=1, head_dim=4, index_topk=3, index_num_heads=0, index_head_dim=0)
    entry = config.build_attention_metadata(_positions([0, 4]))
    metadata = entry.aicpu_metadata
    before = (metadata.sparse_attn.fwd.clone(), metadata.sparse_attn.bwd.clone())

    attention = build_cpu_model(config)

    def consume():
        q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
        out = attention(
            q,
            q.squeeze(2),
            torch.zeros(1),
            entry,
            cmp_k=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
            topk_indices=torch.zeros(1, 4, 1, 3),
            topk_scores=torch.zeros(1, 4, 3, requires_grad=True),
        )
        out.sum().backward()

    for _ in range(3):
        consume()

    # Every kernel call -- three layers' worth of forwards and their backwards -- read the
    # same, still-unchanged metadata tensors.
    assert len(received_fwd) == 3
    assert len(received_bwd) == 3
    assert all(t is metadata.sparse_attn.fwd for t in received_fwd)
    assert all(t is metadata.sparse_attn.bwd for t in received_bwd)
    torch.testing.assert_close(metadata.sparse_attn.fwd, before[0], rtol=0, atol=0)
    torch.testing.assert_close(metadata.sparse_attn.bwd, before[1], rtol=0, atol=0)


def test_the_metadata_flatten_through_the_pytree_registry(monkeypatch):
    """The fused metadata round-trips the compiled path's flattening; the row stays opaque."""
    _fake_metadata_ops(monkeypatch)
    entry = _core_config(compress_ratio=2, num_heads=1, head_dim=4, index_topk=3).build_attention_metadata(
        _positions([0, 4])
    )
    metadata = entry.aicpu_metadata

    leaves, spec = tree_flatten(entry)
    restored = tree_unflatten(leaves, spec)

    # The leaves survive the round trip as the very same objects, and the rebuild keeps the
    # ratio, the frames and the metadata the consumers read off the entry.
    assert isinstance(restored, AscV41Metadata)
    assert restored.ratio == 2
    assert restored.cmp_k is not None and restored.cmp_k.cu_seqlens is entry.cmp_k.cu_seqlens
    assert restored.aicpu_metadata.sparse_attn.fwd is metadata.sparse_attn.fwd
    round_trip, _ = tree_flatten(restored)
    assert len(round_trip) == len(leaves)
    for before, after in zip(leaves, round_trip, strict=True):
        assert before is after
    # The registry flattens past the dataclass and its nodes: the metadata' tensors are leaves in
    # their own right, not opaque blobs.
    assert any(leaf is metadata.sparse_attn.fwd for leaf in leaves)
    assert any(leaf is metadata.sparse_attn.bwd for leaf in leaves)
    assert any(leaf is metadata.lightning_indexer.fwd for leaf in leaves)
    assert any(leaf is metadata.lightning_indexer.bwd for leaf in leaves)
    # The reference row contract is deliberately *not* registered: a stack without the AscendC attention
    # keeps its metadata opaque, so one stack's tracer requirement cannot change another
    # path's flattening.
    row = _reference_row([0, 4], ratio=2)
    row_leaves, _ = tree_flatten(row)
    assert row_leaves == [row]
    # Idempotent: the module-level registrations already ran at import.
    register_pytree_node_for_dataclass(AscV41Metadata)
    register_pytree_node_for_dataclass(KernelFrame)
    register_pytree_node_for_dataclass(AicpuGeometry)
    register_pytree_node_for_dataclass(AicpuMetadata)


def test_the_cp_half_recuts_the_frames_and_rebuilds_the_metadata(monkeypatch):
    """This rank's metadata are built from this rank's frames (upstream #4617's CP half).

    The metadata is the operators' answer for one geometry *of one frame set*: a rank that
    kept the row's would hand the kernels a grid its inputs do not have.  The rebuild runs the
    same builder the microbatch path runs, for the entry's **own** ratio: the queries are this
    rank's chunk of every document, the KV frames stay the whole row, and this geometry's
    compressed axis is that reach divided by its own ratio.
    """
    state = _fake_metadata_ops(monkeypatch)
    config = _core_config(compress_ratio=2, num_heads=1, head_dim=4, index_topk=3)
    row = _positions([0, 16, 40])
    entry = config.build_attention_metadata(row)
    assert len(state["smla_fwd"]) == 1

    cp_size, cp_rank = 4, 2
    mesh = cp_parallel_dims(cp_size, cp_rank).get_mesh("cp")

    local = _prepare_for_rank(entry, mesh, row)

    assert isinstance(local, AscV41Metadata)
    assert local.exchange is not None
    # Chunk queries over the whole-row KV grids: this rank's chunk of every document
    # (``q``), and how far those queries reach into each document (``swa_k``).
    assert local.q.cu_seqlens.tolist() == [0, 4, 10]
    assert local.q.seqused.tolist() == [4, 6]
    assert local.swa_k.cu_seqlens.tolist() == [0, 16, 40]
    assert local.swa_k.seqused.tolist() == [12, 18]
    # The compressed axis is that reach at this geometry's own ratio -- not the row's frame,
    # and not a grid this rank divided for itself.
    assert local.cmp_k is not None
    assert local.cmp_k.cu_seqlens.tolist() == [0, 8, 20]
    assert local.cmp_k.seqused.tolist() == [6, 9]
    assert local.cmp_k.cu_seqlens is entry.cu_seq_k
    torch.testing.assert_close(local.positions_q, local.exchange.positions_q.reshape(1, -1), rtol=0, atol=0)
    assert local.q.cu_seqlens.tolist() != entry.q.cu_seqlens.tolist()
    # The metadata are rebuilt from the rank-local frames, never shared with the row's.
    rank_metadata, row_metadata = local.aicpu_metadata, entry.aicpu_metadata
    assert rank_metadata.sparse_attn.fwd is not row_metadata.sparse_attn.fwd
    assert rank_metadata.sparse_attn.bwd is not row_metadata.sparse_attn.bwd
    assert rank_metadata.lightning_indexer.fwd is not row_metadata.lightning_indexer.fwd
    assert rank_metadata.lightning_indexer.bwd is not row_metadata.lightning_indexer.bwd
    assert len(state["smla_fwd"]) == 2
    assert len(state["smla_bwd"]) == 2
    assert len(state["li"]) == 2
    assert len(state["slig"]) == 2
    # The geometry rides along, so a second transform would rebuild from the same values.
    assert local.aicpu_metadata.geometry == entry.aicpu_metadata.geometry
    assert local.ratio == entry.ratio == 2


def test_the_cp_alignment_is_this_geometries_own_factor(monkeypatch):
    """Each geometry checks its own ``cp_size * ratio``; the stack's lcm is the loader's.

    A document must be a whole number of chunks and of pooling groups, so a length that is not
    a multiple of ``cp_size * ratio`` cannot be cut: the first factor makes every prefix a
    whole number of chunks, the second keeps every compressed prefix a whole number of groups.
    The stack-wide lcm is the dataloader's alignment, so what one entry enforces is its own
    factor and the entries together are the same constraint.
    """
    _fake_metadata_ops(monkeypatch)
    aligned = _core_config(compress_ratio=2, index_num_heads=0, index_head_dim=0).build_attention_metadata(
        _positions([0, 8, 16])
    )
    sliced = shard_asc_v41_metadata(aligned, cp_size=4, cp_rank=0, batch_size=1, seq_len=4)
    assert sliced.q.cu_seqlens.tolist() == [0, 2, 4]

    ragged = _core_config(compress_ratio=2, index_num_heads=0, index_head_dim=0).build_attention_metadata(
        _positions([0, 6, 12])
    )
    with pytest.raises(ValueError, match=r"cp_size \* compress_ratio = 8"):
        shard_asc_v41_metadata(ragged, cp_size=4, cp_rank=0, batch_size=1, seq_len=3)


def test_the_seam_transforms_the_map_under_context_parallel(monkeypatch):
    """End to end: the framework slices the inputs and every geometry's entry follows them.

    The model no longer shards its own inputs -- ``prepare_context_parallel_input`` does, one
    even slab per rank when no load balancer is configured -- and the metadata a rank reads is
    prepared *after* that shard, from the pre-shard positions, **per geometry**: each entry is
    rebuilt from its own ratio's frames rather than one stack-wide object being re-cut once.
    """
    state = _fake_metadata_ops(monkeypatch)
    install_plain_cp_shard(monkeypatch)
    row = _positions([0, 8, 24])
    tokens = torch.zeros_like(row)
    cp_size, cp_rank = 2, 1
    _, _, extra = preprocess(
        _fused_tiny_model(monkeypatch),
        tokens,
        tokens,
        row,
        parallel_dims=cp_parallel_dims(cp_size, cp_rank),
    )

    slab = row.shape[-1] // cp_size
    assert extra["positions"].shape[-1] == slab
    assert extra["positions"].tolist() == row[:, cp_rank * slab : (cp_rank + 1) * slab].tolist()
    metadata = extra["attention_metadata"]
    assert set(metadata) == {
        (AscV41SparseAttention.Config, 0),
        (AscV41SparseAttention.Config, 1),
        (AscV41SparseAttention.Config, 2),
    }
    for (port, ratio), entry in metadata.items():
        assert port is AscV41SparseAttention.Config
        assert entry.ratio == ratio
        assert entry.exchange is not None
        assert entry.q.cu_seqlens.tolist() == [0, 4, 12]
        assert entry.swa_k.seqused.tolist() == [8, 16]
    # One geometry has no compressed axis at all, the other two carry their own.
    assert metadata[(AscV41SparseAttention.Config, 0)].cmp_k is None
    assert metadata[(AscV41SparseAttention.Config, 1)].cmp_k.seqused.tolist() == [8, 16]
    assert metadata[(AscV41SparseAttention.Config, 2)].cmp_k.seqused.tolist() == [4, 8]
    # One build for the row and one rebuild for this rank's frames, per geometry.
    assert len(state["smla_fwd"]) == 6
    assert len(state["li"]) == 4


def test_the_cp_half_needs_the_mesh_and_the_positions(monkeypatch):
    """A transform without the geometry it derives this rank's copy from fails loudly."""
    _fake_metadata_ops(monkeypatch)
    entry = _core_config(compress_ratio=2, index_num_heads=0, index_head_dim=0).build_attention_metadata(
        _positions([0, 4])
    )
    mesh = cp_parallel_dims(2, 0).get_mesh("cp")
    with pytest.raises(ValueError, match="needs the CP mesh"):
        _prepare_for_rank(entry, mesh, None)
    with pytest.raises(ValueError, match="needs the CP mesh"):
        _prepare_for_rank(entry, None, _positions([0, 4]))


def test_the_override_factory_swaps_the_core_and_its_builder(monkeypatch):
    """``asc`` installs the port: the class, its key, and a builder that precomputes.

    The five numbers are the recipe's to state (the defaults are the released shape, pinned in
    ``test_fused_geometry``); what this file pins is that the swap carries them onto the *port's*
    config and that the config which follows the class is the one whose builder sizes the metadata.
    """
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn import asc

    state = _fake_metadata_ops(monkeypatch)
    base = CompressedSparseInnerAttention2.Config(window_size=2, softmax_scale=0.5, compress_ratio=2)

    port = asc(base, num_heads=1, head_dim=4, index_topk=3, index_num_heads=1, index_head_dim=4)

    assert type(port) is AscV41SparseAttention.Config
    assert isinstance(port.build(), AscV41SparseAttention)
    # The reference core's fields survive the swap, and the key names the swapped class.
    assert (port.window_size, port.softmax_scale, port.compress_ratio) == (2, 0.5, 2)
    assert port.attention_metadata_key == (AscV41SparseAttention.Config, 2)
    # The builder follows the class: this config now precomputes this geometry's metadata.
    assert type(port).build_attention_metadata is AscV41SparseAttention.Config.build_attention_metadata
    built = port.build_attention_metadata(_positions([0, 4]))
    assert built.aicpu_metadata.geometry == port.geometry
    assert built.cmp_k is not None and built.cmp_k.cu_seqlens.tolist() == [0, 2]
    assert [len(state[kind]) for kind in ("smla_fwd", "smla_bwd", "li", "slig")] == [1, 1, 1, 1]


def _fuse_every_layer(config):
    """Swap the AscendC attention in on every layer, stating the geometry its kernels are sized with.

    The override targets the core config with ``exact=True``, so a real run reaches all 40
    layers; the geometry is the tiny debug flavor's own numbers, which is what the recipe's
    override arguments carry.
    """
    for layer in config.layers:
        layer.attention.inner_attention = derive(
            layer.attention.inner_attention,
            AscV41SparseAttention.Config,
            **_TINY_GEOMETRY,
        )


def _fused_tiny_model(monkeypatch):
    """The tiny debug flavor with the AscendC attention on every layer, as the recipe installs it."""
    return tiny_v41_model(monkeypatch, configure=_fuse_every_layer)[0]


def _prepare_for_rank(attention_metadata, mesh, positions):
    """Run one entry's CP transform the way ``Decoder._cp_shard`` does.

    The transform takes the permutation alone (upstream #4617); the mesh and the *pre-shard*
    row come from the context the patched shard publishes around it.  It is the metadata's own
    hook, which is what lets a map key be a geometry rather than a backend class.
    """
    with cp_shard_context_set(cp_mesh=mesh, load_balancer_type=None, positions=positions):
        return AscV41Metadata.prepare_cp_metadata(attention_metadata, permutation=None)
