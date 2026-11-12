# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract of DeepSeek-V4's backend-owned metadata seam.

The sparse-attention core's own config builds what it consumes (upstream #4897): the
reference tier the packed contract, the fused AscendC tier that contract plus this
geometry's kernel records.  The decoder builds one entry per backend geometry for the
whole microbatch, and the fused entry's own ``prepare_cp_metadata`` re-derives this
rank's copy (upstream #4617).  The test keeps the contract's two ends -- per-geometry
entries and the rank-local rebuild -- and drops the per-field detail.
"""

import pytest
import torch

from torchtitan_npu.models.common.restore import DOC_CHUNK_LAYOUT
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.backend import with_npu_attention_backend
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.models.deepseek_v4.sharding import set_deepseek_v4_sharding_config
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4.metadata import AicpuGeometry, AscV4Metadata
from torchtitan_npu.override.deepseek_v4.sparse_attn import asc_legacy
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscCompressedSparseInnerAttention
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set

pytestmark = pytest.mark.usefixtures("dsv4")

_DIM, _HEADS, _HEAD_DIM = 32, 4, 8
_INDEX_N_HEADS, _INDEX_HEAD_DIM, _INDEX_TOPK, _WINDOW = 2, 8, 16, 128


def _positions_for(doc_lens) -> torch.Tensor:
    return torch.cat([torch.arange(length) for length in doc_lens])


def _model_config(ratios: tuple[int, ...], *, ascendc: bool, geometry: dict | None = None):
    config = _make_v4_config(
        dim=_DIM,
        n_layers=len(ratios),
        vocab_size=64,
        n_heads=_HEADS,
        head_dim=_HEAD_DIM,
        rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=ratios,
        window_size=_WINDOW,
        norm_eps=1e-6,
        index_n_heads=_INDEX_N_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
        index_topk=_INDEX_TOPK,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=_WINDOW * 2,
        compress_rope_theta=10000.0,
        original_seq_len=_WINDOW * 2,
        num_mtp_layers=0,
    )
    # These tests exercise the explicitly retained geometry/carry backend.
    config = with_npu_attention_backend(config)
    # The sharding pass runs at config resolution, before the overrides.
    set_deepseek_v4_sharding_config(config, enable_sp=False, enable_ep=False)
    if ascendc:
        for _, inner, parent, attr in list(config.traverse(CompressedSparseInnerAttention.Config)):
            setattr(
                parent,
                attr,
                asc_legacy(
                    inner,
                    **(
                        geometry
                        or {
                            "num_heads": _HEADS,
                            "head_dim": _HEAD_DIM,
                            "index_num_heads": _INDEX_N_HEADS,
                            "index_head_dim": _INDEX_HEAD_DIM,
                        }
                    ),
                ),
            )
    return config


def _entry(model, ratio: int = 4, positions=None):
    metadata = model._get_attention_metadata(_positions_for((_WINDOW, _WINDOW)) if positions is None else positions)
    matches = [entry for entry in metadata.values() if entry.ratio == ratio]
    assert len(matches) == 1
    return matches[0]


def test_native_model_rejects_cp_even_with_aligned_documents():
    from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()
    config.parallelism.context_parallel_degree = 2
    config.dataloader.dataset.dataset.processor.per_doc_alignment = 256

    with pytest.raises(NotImplementedError, match="TP=CP=PP=1"):
        config.model.update_from_config(config=config)


def test_entries_are_built_per_geometry():
    """One entry per (backend class, ratio), and every layer dispatches by its own key."""
    model = _model_config((1, 1, 4, 128), ascendc=False).build()

    metadata = model._get_attention_metadata(_positions_for((128, 128)))

    assert {ratio for _, ratio in metadata} == {1, 4, 128}
    entry = metadata[(CompressedSparseInnerAttention.Config, 4)]
    assert type(entry) is DeepSeekV4Metadata and entry.ratio == 4
    # The row grid lives on the sliding-window frame; the pool grid is this entry's own fold of it,
    # and an unpooled geometry has no pool at all.  At one degree nothing is moved or restored.
    assert entry.swa_k.cu_seqlens.tolist() == [0, 128, 256]
    assert entry.cmp_k is not None and entry.cmp_k.cu_seqlens.tolist() == [0, 32, 64]
    assert metadata[(CompressedSparseInnerAttention.Config, 1)].cmp_k is None
    assert entry.cp is None
    keys = {layer.attention.attention_metadata_key for layer in model.layers.values()}
    assert {key[1] for key in keys} == {1, 4, 128}
    assert all(metadata[key].ratio == key[1] for key in keys)


@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_attention_without_cp_matches_independent_samples(batch_size, monkeypatch):
    """No-CP attention must preserve independent document rows and their gradients.

    The folded row packs what used to be batch samples as documents, so the
    oracle applies the same initialized layer to each document separately.
    """
    seq_len = 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = _model_config((1,), ascendc=True).build()
        attention = model.layers["0"].attention
        attention.init_states(buffer_device=torch.device("cpu"))

    def cpu_window_attention(q, swa_k, attn_sink, *, attention_metadata, **kwargs):
        # The fused kernel is unavailable on CPU, while attn-gym's packed reference
        # addresses one row. Isolate only that kernel with per-document dense
        # causal math; RoPE and all input/output projections remain production code.
        del kwargs
        assert attention_metadata.cp is None
        cu = attention_metadata.q.cu_seqlens.to(torch.long).tolist()
        total = q.size(0)
        scores = torch.einsum("thd,sd->ths", q.float(), swa_k.float()) * attention.inner_attention.softmax_scale
        allow = torch.zeros(total, total, dtype=torch.bool)
        for start, end in zip(cu[:-1], cu[1:], strict=False):
            allow[start:end, start:end] = torch.ones(end - start, end - start, dtype=torch.bool).tril()
        scores = scores.masked_fill(~allow[:, None, :], float("-inf"))
        sink = attn_sink[None, :, None].expand(*scores.shape[:-1], 1)
        probabilities = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
        return torch.einsum("ths,sd->thd", probabilities, swa_k)

    monkeypatch.setattr(attention.inner_attention, "forward", cpu_window_attention)
    values = torch.linspace(-1.0, 1.0, batch_size * seq_len * _DIM).reshape(batch_size * seq_len, _DIM)
    x = values.clone().requires_grad_()
    positions = _positions_for((seq_len,) * batch_size)
    entry = model._get_attention_metadata(positions)[attention.attention_metadata_key]
    # The folded row is one packed row: ``batch_size`` is its own count, not the
    # number of documents that were packed into it.
    assert entry.batch_size == 1
    # At one degree the query frame is the row's own grid and every document is a
    # batch sample here; the KV frame addresses that same row.
    assert entry.q.cu_seqlens.tolist() == [row * seq_len for row in range(batch_size + 1)]
    assert entry.swa_k.cu_seqlens.tolist() == entry.q.cu_seqlens.tolist()
    parameters = tuple(attention.parameters())
    upstream = torch.linspace(0.25, 1.25, x.numel()).reshape_as(x)

    output = attention(x, entry, positions)
    assert output.shape == (batch_size * seq_len, _DIM)
    gradients = torch.autograd.grad((output * upstream).sum(), (x, *parameters))

    # Distinct documents have no cross-row attention. Applying the same initialized
    # layer to each document separately supplies the batch-composition oracle, including
    # the sum of the individual parameter gradients for the same scalar objective.
    expected_outputs, expected_input_gradients = [], []
    expected_parameter_gradients = [torch.zeros_like(parameter) for parameter in parameters]
    for row in range(batch_size):
        slab = slice(row * seq_len, (row + 1) * seq_len)
        sample = values[slab].clone().requires_grad_()
        sample_positions = _positions_for((seq_len,))
        sample_entry = model._get_attention_metadata(sample_positions)[attention.attention_metadata_key]
        expected = attention(sample, sample_entry, sample_positions)
        sample_gradients = torch.autograd.grad((expected * upstream[slab]).sum(), (sample, *parameters))
        expected_outputs.append(expected)
        expected_input_gradients.append(sample_gradients[0])
        for accumulated, gradient in zip(expected_parameter_gradients, sample_gradients[1:], strict=True):
            accumulated.add_(gradient)

    torch.testing.assert_close(output, torch.cat(expected_outputs), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradients[0], torch.cat(expected_input_gradients), rtol=1e-5, atol=1e-6)
    for actual, expected in zip(gradients[1:], expected_parameter_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_the_fused_entry_carries_its_geometry_and_records(dsv4):
    """The fused core's config builds the packed contract and its kernel records."""
    dsv4.cann_ops.calls.clear()
    model = _model_config((1, 4, 128), ascendc=True).build()

    metadata = model._get_attention_metadata(_positions_for((128, 128, 128, 128)))

    geometry = AicpuGeometry(
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        window_size=_WINDOW,
        index_topk=_INDEX_TOPK,
        index_num_heads=_INDEX_N_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
    )
    for (cls, ratio), entry in metadata.items():
        assert cls is AscCompressedSparseInnerAttention.Config and entry.ratio == ratio
        assert isinstance(entry, AscV4Metadata)
        assert entry.aicpu_metadata.geometry == geometry
        assert entry.aicpu_metadata.sparse_attn.fwd is not None
        assert entry.aicpu_metadata.sparse_attn.bwd is not None
        assert entry.q is not None and entry.swa_k is not None
        assert entry.cmp_k is None if ratio == 1 else entry.cmp_k is not None
        # The fused core never reads ``slots``: its builder skips the reference-only
        # S x (S / ratio) HCA matrix even for the ratio-128 geometry.
        assert entry.slots is None
        if ratio == 4:
            assert entry.aicpu_metadata.lightning_indexer.fwd is not None
        else:
            assert entry.aicpu_metadata.lightning_indexer.fwd is None
    smla = [call for call in dsv4.cann_ops.calls if call[0] == "sparse_flash_mla_metadata"]
    assert len(smla) == 3
    assert {call[2]["cmp_ratio"] for call in smla} == {1, 4, 128}


def test_only_the_reference_builds_the_hca_selection():
    """``slots`` is the reference core's precomputed selection, not the fused core's.

    The reference forward reads it (``kv_indices`` for a geometry without a selector); the
    fused core's HCA branch runs full-causal on the compressed pool and never touches it, so
    its builder passes ``build_slots=False`` and the ``S x (S / ratio)`` int64 matrix -- 64 GiB
    at S=1M, ratio=128 -- is never materialised on the fused path.
    """
    positions = _positions_for((128, 128))
    reference = _entry(_model_config((128,), ascendc=False).build(), 128, positions)
    assert reference.slots is not None
    assert reference.slots.shape == (1, 256, 2)

    fused = _entry(_model_config((128,), ascendc=True).build(), 128, positions)
    assert fused.slots is None


class _FakeCPMesh:
    """The 1-D mesh contract the metadata's CP transform reads."""

    ndim = 1
    device_type = "cpu"

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_local_rank(self) -> int:
        return self._rank


def _prepare_for_rank(attention_metadata, positions, *, rank: int, cp_size: int = 2):
    with cp_shard_context_set(
        cp_mesh=_FakeCPMesh(rank, cp_size),
        load_balancer_type=DOC_CHUNK_LAYOUT,
        positions=positions,
    ):
        return type(attention_metadata).prepare_cp_metadata(attention_metadata, permutation=None)


def _prefix(lengths: torch.Tensor) -> torch.Tensor:
    """The frame of a per-document prefix: the running sum of the reach, from zero."""
    return torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0).to(torch.int32)])


def test_the_fused_cp_transform_rebuilds_rank_local_frames(dsv4, monkeypatch):
    """The CP half rebuilds this rank's frames and records from the global row."""
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    built: list[torch.Tensor] = []

    def recording_smla_metadata(*args, **kwargs):
        built.append(kwargs["cu_seqlens_q"])
        return torch.tensor([kwargs["cu_seqlens_q"][-1].item()], dtype=torch.int32)

    monkeypatch.setattr(ascendc, "sparse_flash_mla_metadata", recording_smla_metadata)
    positions = _positions_for((_WINDOW, _WINDOW // 2))
    model = _model_config((1, 4), ascendc=True).build()

    global_metadata = _entry(model, 4, positions)
    row = torch.tensor([0, _WINDOW, _WINDOW + _WINDOW // 2], dtype=torch.int32)
    chunk = torch.diff(row) // 2

    rank_metadata = [_prepare_for_rank(global_metadata, positions, rank=rank) for rank in range(2)]
    for rank, metadata in enumerate(rank_metadata):
        # The row stays the whole row; the queries are this rank's chunks.
        _prefix_of_reach = _prefix((rank + 1) * chunk)
        assert metadata.cmp_k is not None
        assert torch.equal(metadata.q.cu_seqlens, (row // 2).to(torch.int32))
        assert torch.equal(metadata.q.seqused, chunk)
        # The key frames are the ones the kernels read: the row itself where they honour
        # ``seqused``, the reach where they anchor their mask at the frame's end.
        expected_window = row if _IS_A5 else _prefix_of_reach
        assert torch.equal(metadata.swa_k.cu_seqlens, expected_window)
        assert torch.equal(metadata.swa_k.seqused, (rank + 1) * chunk)
        expected_pool = (row // 4).to(torch.int32) if _IS_A5 else _prefix_of_reach // 4
        assert torch.equal(metadata.cmp_k.cu_seqlens, expected_pool)
        assert torch.equal(metadata.cmp_k.seqused, ((rank + 1) * chunk) // 4)
        assert metadata.ratio == 4
        assert metadata.aicpu_metadata.sparse_attn.fwd is not global_metadata.aicpu_metadata.sparse_attn.fwd
        assert metadata.aicpu_metadata.geometry == global_metadata.aicpu_metadata.geometry
        assert metadata.cp is not None and metadata.cp.restore.exchange is not None
    # The restore is rank-dependent even though the row is.
    assert rank_metadata[0].position_k.tolist() != rank_metadata[1].position_k.tolist()


def test_the_inner_attention_and_selector_own_the_cp_boundary():
    """The core gathers its KV streams; the selector gathers the index keys it scores.

    Both conversions are the V4.1 ``S(1) -> R`` pattern, declared by the modules the layer
    calls directly (there is no wrapper between them any more).
    """
    config = _model_config((4,), ascendc=True)
    attention = config.layers[0].attention
    core_sharding = attention.inner_attention.sharding_config
    assert core_sharding is not None
    names = set(core_sharding.in_src_shardings)
    assert not names & {"idx_q", "idx_k", "idx_w"}
    assert {"q", "swa_k", "cmp_k", "topk_indices", "topk_scores", "attn_sink"} <= names
    # Positional forward args are (q, swa_k, attn_sink): the KV streams are gathered on the
    # way in, so their destination placement is the whole row.
    core_row = core_sharding.in_dst_shardings
    assert core_row["swa_k"] != core_sharding.in_src_shardings["swa_k"]
    assert core_row["cmp_k"] != core_sharding.in_src_shardings["cmp_k"]

    selector_sharding = attention.lightning_indexer.sharding_config
    assert selector_sharding is not None
    assert set(selector_sharding.in_src_shardings) == {"idx_q", "idx_k", "idx_w"}
    assert selector_sharding.in_dst_shardings["idx_k"] != selector_sharding.in_src_shardings["idx_k"]
