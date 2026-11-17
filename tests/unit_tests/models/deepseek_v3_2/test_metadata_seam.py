# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract of DeepSeek-V3.2's move onto the backend-owned metadata seam.

The stack used to build one mask object in ``DeepSeekV32Model`` and post-process
it through a ``metadata_extension``.  Now every attention backend declares what
it consumes (upstream #4897's ``build_attention_metadata``): the reference core
names the packed layout it reads, the AscendC override wraps that layout with the
host length lists its TND kernels take, and the decoder builds one entry per
backend for the whole microbatch.  These tests pin that sharing, the CP half the
AscendC entry derives for itself, and the reference core's refusal to be sharded.
"""

from types import SimpleNamespace

import pytest
import torch

import torchtitan_npu  # noqa: F401  imported for its patch chain: the seam under test
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan_npu.models.deepseek_v3_2 import (
    ComplexRoPE,
    DeepSeekV32Model,
    Embedding,
    Linear,
    RMSNorm,
    _build_dsv3_2_layers,
)
from torchtitan_npu.models.deepseek_v3_2 import model as v32_model
from torchtitan_npu.models.deepseek_v3_2.model import ReferenceVarlenMetadata, SparseInnerAttention
from torchtitan_npu.override.deepseek_v3_2.sparse_attn import asc
from torchtitan_npu.override.deepseek_v3_2.sparse_attn.ascendc import AscSparseInnerAttention, AscVarlenMetadata
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set

# Two documents of four tokens in one row: the headtail load balancer needs a
# sequence length divisible by ``2 * cp`` for the two-rank case below.
_POSITIONS = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
_SEQ_LEN = 8
_CU_SEQ = [0, 4, 8]
_NO_CP_DIMS = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None)
_NO_SPMD = SimpleNamespace(spmd_backend="partial_dtensor")


class _FakeCPMesh:
    """The 1-D mesh contract ``CPVarlenMetadata.from_global`` reads."""

    ndim = 1
    device_type = "cpu"

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_local_rank(self) -> int:
        return self._rank


def _model_config(n_layers: int = 3, *, ascendc: bool = False) -> DeepSeekV32Model.Config:
    layers = _build_dsv3_2_layers(
        n_layers=n_layers,
        n_dense_layers=n_layers,
        dim=32,
        n_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        mscale=1.0,
        dense_hidden_dim=64,
        moe_hidden_dim=16,
        num_experts=4,
        num_shared_experts=1,
        router_top_k=2,
        router_score_func="softmax",
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        rope=ComplexRoPE.Config(dim=8, max_seq_len=_SEQ_LEN, theta=10000.0, original_seq_len=_SEQ_LEN),
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
    )
    config = DeepSeekV32Model.Config(
        vocab_size=64,
        dim=32,
        tok_embeddings=Embedding.Config(num_embeddings=64, embedding_dim=32),
        norm=RMSNorm.Config(normalized_shape=32),
        lm_head=Linear.Config(in_features=32, out_features=64),
        layers=layers,
    )
    if ascendc:
        # What ``--override.imports ...sparse_attn.asc`` does to a recipe.
        for _, inner, parent, attr in list(config.traverse(SparseInnerAttention.Config)):
            setattr(parent, attr, asc(inner))
    return config


@pytest.fixture
def builds(monkeypatch) -> list:
    """Record every packed-layout build, whichever backend asked for one."""
    recorded: list = []
    build = v32_model.create_varlen_metadata_for_document

    def counting_build(positions):
        recorded.append(positions)
        return build(positions)

    monkeypatch.setattr(v32_model, "create_varlen_metadata_for_document", counting_build)
    return recorded


def test_the_decoder_builds_one_entry_for_the_whole_stack(builds):
    """Three layers, one geometry: the microbatch pays for the layout once."""
    model = _model_config(n_layers=3).build()

    attention_metadata = model._get_attention_metadata(_POSITIONS)

    assert len(builds) == 1
    assert set(attention_metadata) == {SparseInnerAttention}
    entry = attention_metadata[SparseInnerAttention]
    assert isinstance(entry, ReferenceVarlenMetadata)
    assert entry.cu_seq_q.tolist() == _CU_SEQ
    # Every layer dispatches by the key the one entry is stored under.
    keys = {layer.attention.attention_metadata_key for layer in model.layers.values()}
    assert keys == {SparseInnerAttention}
    assert all(attention_metadata[key] is entry for key in keys)


def test_preprocess_inputs_hands_the_map_to_the_forward():
    """The model's own preprocessing yields the map, not a single mask object."""
    model = _model_config(n_layers=1).build()

    _, _, extra_kwargs = model.preprocess_inputs(
        {"input": torch.zeros(1, _SEQ_LEN, dtype=torch.int64), "positions": _POSITIONS},
        torch.zeros(1, _SEQ_LEN, dtype=torch.int64),
        parallel_dims=_NO_CP_DIMS,
        device=torch.device("cpu"),
        parallelism=_NO_SPMD,
    )

    assert set(extra_kwargs["attention_metadata"]) == {SparseInnerAttention}
    assert "attention_masks" not in extra_kwargs


def test_the_ascendc_backend_owns_the_tnd_metadata(builds):
    """The fused backend wraps the shared layout with its host length lists."""
    model = _model_config(n_layers=3, ascendc=True).build()

    attention_metadata = model._get_attention_metadata(_POSITIONS)

    assert len(builds) == 1
    assert set(attention_metadata) == {AscSparseInnerAttention}
    entry = attention_metadata[AscSparseInnerAttention]
    assert isinstance(entry, AscVarlenMetadata)
    # The TND kernels take the cumulative end offsets on host, not the device
    # tensor they were read from.
    assert entry.actual_seq_qlen == _CU_SEQ[1:]
    assert entry.actual_seq_klen == _CU_SEQ[1:]
    assert isinstance(entry.varlen_meta, VarlenMetadata)
    assert entry.varlen_meta.cu_seq_q.tolist() == _CU_SEQ
    assert {layer.attention.attention_metadata_key for layer in model.layers.values()} == {AscSparseInnerAttention}


def test_the_ascendc_entry_prepares_this_ranks_copy_for_context_parallel():
    """Under CP the entry re-reads its host lists from the rank-local offsets."""
    global_entry = AscVarlenMetadata.from_varlen(v32_model.build_reference_varlen_metadata(_POSITIONS))
    assert global_entry.actual_seq_qlen == _CU_SEQ[1:]

    locals_per_rank = [_prepare_for_rank(AscVarlenMetadata, global_entry, rank) for rank in range(2)]

    for entry in locals_per_rank:
        assert isinstance(entry.varlen_meta, CPVarlenMetadata)
        # Each rank holds half the row, and its host lists describe that half:
        # the offsets are this rank's own, not the global row's.
        assert entry.actual_seq_qlen == entry.varlen_meta.cu_seq_q[1:].cpu().tolist()
        assert entry.actual_seq_qlen[-1] == _SEQ_LEN // 2
        assert entry.actual_seq_qlen != global_entry.actual_seq_qlen
        assert entry.varlen_meta.k_global_gather_indices.numel() > 0

    # The two ranks' segments differ: the copy really is rank-local.
    assert locals_per_rank[0].varlen_meta.k_global_gather_indices.tolist() != (
        locals_per_rank[1].varlen_meta.k_global_gather_indices.tolist()
    )


def test_the_reference_core_refuses_context_parallel_while_preparing():
    """A stack that cannot be sharded says so during preparation, not later."""
    entry = v32_model.build_reference_varlen_metadata(_POSITIONS)

    with pytest.raises(ValueError, match="not context-parallel"):
        _prepare_for_rank(ReferenceVarlenMetadata, entry, 0)


def _prepare_for_rank(metadata_type, attention_metadata, rank: int):
    """Run one metadata type's CP transform the way ``Decoder._cp_shard`` does.

    The transform takes the permutation alone (upstream #4617); what it cannot
    recover -- the mesh, the load-balancer type and the *pre-shard* positions --
    comes from the context the patched shard publishes around the call.
    """
    with cp_shard_context_set(
        cp_mesh=_FakeCPMesh(rank, 2),
        load_balancer_type="headtail",
        positions=_POSITIONS,
    ):
        return metadata_type.prepare_cp_metadata(attention_metadata, permutation=None)
