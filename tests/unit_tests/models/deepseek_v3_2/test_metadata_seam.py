# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract of DeepSeek-V3.2's move onto the backend-owned metadata seam.

The stack used to build one mask object in ``DeepSeekV32Model`` and post-process
it through a ``metadata_extension``.  Now every attention backend declares what
it consumes (upstream #4897's ``build_attention_metadata``): the reference core
names the packed layout it reads, the AscendC override wraps that layout with the
host length list its TND kernels take, and the decoder builds one entry per
backend for the whole microbatch.  These tests pin that sharing and the AscendC
entry's host length list.
"""

from types import SimpleNamespace

import pytest
import torch
from torchtitan.config import ParallelismConfig
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.common.decoder import Decoder

import torchtitan_npu  # noqa: F401  imported for its patch chain: the seam under test
from torchtitan_npu.models.deepseek_v3_2 import (
    ComplexRoPE,
    DeepSeekV32Model,
    Embedding,
    Linear,
    RMSNorm,
    _build_dsv3_2_layers,
)
from torchtitan_npu.models.deepseek_v3_2 import model as v32_model
from torchtitan_npu.models.deepseek_v3_2.model import SparseInnerAttention
from torchtitan_npu.override.deepseek_v3_2.sparse_attn import asc, ascendc
from torchtitan_npu.override.deepseek_v3_2.sparse_attn.ascendc import AscSparseInnerAttention, AscVarlenMetadata

# Two documents of four tokens in one packed row.
_POSITIONS = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
_SEQ_LEN = 8
_CU_SEQ = [0, 4, 8]
_NO_CP_DIMS = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None)
_NO_SPMD = SimpleNamespace(spmd_backend="partial_dtensor")


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
        rope=ComplexRoPE.Config(dim=8, max_context_length=_SEQ_LEN, theta=10000.0, original_seq_len=_SEQ_LEN),
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
    """Record every packed-layout build, whichever backend asked for one.

    Both backends capture ``create_varlen_metadata_for_document`` in their own
    module namespace, so the counter is installed in each of the two.
    """
    recorded: list = []
    build = v32_model.create_varlen_metadata_for_document

    def counting_build(positions):
        recorded.append(positions)
        return build(positions)

    monkeypatch.setattr(v32_model, "create_varlen_metadata_for_document", counting_build)
    monkeypatch.setattr(ascendc, "create_varlen_metadata_for_document", counting_build)
    return recorded


def test_the_decoder_builds_one_entry_for_the_whole_stack(builds):
    """Three layers, one geometry: the microbatch pays for the layout once."""
    model = _model_config(n_layers=3).build()

    attention_metadata = model._get_attention_metadata(_POSITIONS)

    assert len(builds) == 1
    assert set(attention_metadata) == {SparseInnerAttention}
    entry = attention_metadata[SparseInnerAttention]
    assert isinstance(entry, VarlenMetadata)
    assert entry.cu_seq_q.tolist() == _CU_SEQ
    # Every layer dispatches by the key the one entry is stored under.
    keys = {layer.attention.attention_metadata_key for layer in model.layers.values()}
    assert keys == {SparseInnerAttention}
    assert all(attention_metadata[key] is entry for key in keys)


def test_preprocess_inputs_hands_the_map_to_the_forward():
    """The model's own preprocessing yields the map, not a single mask object."""
    model = _model_config(n_layers=1).build()

    _, _, extra_kwargs = model.preprocess_inputs(
        {
            "input": torch.zeros(_SEQ_LEN, dtype=torch.int64),
            "labels": torch.zeros(_SEQ_LEN, dtype=torch.int64),
            "positions": _POSITIONS,
        },
        parallel_dims=_NO_CP_DIMS,
        parallelism=_NO_SPMD,
    )

    assert set(extra_kwargs["attention_metadata"]) == {SparseInnerAttention}
    assert "attention_masks" not in extra_kwargs


def test_decoder_update_from_config_refuses_context_parallel():
    """The decoder guard the V3.2 config delegates to refuses CP for these cores.

    This is the guard the retired rank-local hook used to supplement: it runs at
    configuration time, before any layer could read a global varlen layout.  The
    test calls the decoder contract directly so it does not depend on the
    deferred pre-fold ``DeepSeekV32Model.Config.update_from_config`` rope step.
    """
    config = _model_config(n_layers=1)

    with pytest.raises(NotImplementedError, match="Context Parallel is not supported"):
        Decoder.Config.update_from_config(
            config,
            config=SimpleNamespace(
                parallelism=ParallelismConfig(
                    context_parallel_degree=2,
                    spmd_backend="spmd_types",
                ),
            ),
        )


def test_the_ascendc_backend_owns_the_tnd_metadata(builds):
    """The fused backend wraps the shared layout with its host length list."""
    model = _model_config(n_layers=3, ascendc=True).build()

    attention_metadata = model._get_attention_metadata(_POSITIONS)

    assert len(builds) == 1
    assert set(attention_metadata) == {AscSparseInnerAttention}
    entry = attention_metadata[AscSparseInnerAttention]
    assert isinstance(entry, AscVarlenMetadata)
    # The TND kernels take the cumulative end offsets on host, not the device
    # tensor they were read from.  One list serves the q and k kernel arguments
    # because the builder hands both offset tensors the same packed layout.
    assert entry.actual_seq_lengths == _CU_SEQ[1:]
    assert isinstance(entry.varlen_meta, VarlenMetadata)
    assert entry.varlen_meta.cu_seq_q.tolist() == _CU_SEQ
    assert entry.varlen_meta.cu_seq_k.tolist() == _CU_SEQ
    assert {layer.attention.attention_metadata_key for layer in model.layers.values()} == {AscSparseInnerAttention}
