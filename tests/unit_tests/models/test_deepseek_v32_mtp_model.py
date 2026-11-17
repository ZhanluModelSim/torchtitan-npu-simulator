# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.unit_tests.models.mtp_test_utils import (
    BATCH_SIZE,
    SEQ_LEN,
    VOCAB_SIZE,
    assert_packed_mtp_training,
    build_cpu_model,
)
from torchtitan_npu.models.deepseek_v3_2 import (
    ComplexRoPE,
    DeepSeekV32Model,
    Embedding,
    Linear,
    RMSNorm,
    _build_dsv3_2_layers,
    _build_mtp_layers,
)
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss


def _build_rope_config() -> ComplexRoPE.Config:
    return ComplexRoPE.Config(
        dim=8,
        max_seq_len=SEQ_LEN,
        theta=10000.0,
        original_seq_len=SEQ_LEN,
    )


def _build_model_config():
    dim = 32
    layers = _build_dsv3_2_layers(
        n_layers=1,
        n_dense_layers=1,
        dim=dim,
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
        rope=_build_rope_config(),
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
    )
    return DeepSeekV32Model.Config(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=dim,
        ),
        norm=RMSNorm.Config(normalized_shape=dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=VOCAB_SIZE,
        ),
        layers=layers,
        mtp_layers=_build_mtp_layers(layers[-1], dim=dim, num_mtp_layers=1),
    )


@pytest.fixture(autouse=True)
def _step_denominator():
    """The trainer sets the step's global valid-token count; the indexer loss normalizes by it."""
    AuxLoss.set_step_denominator(torch.tensor(BATCH_SIZE * SEQ_LEN, dtype=torch.float32))
    yield
    AuxLoss._step_denominator = None


def test_deepseek_v32_mtp_model_supports_packed_sequences():
    assert_packed_mtp_training(build_cpu_model(_build_model_config()))
