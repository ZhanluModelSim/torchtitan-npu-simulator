# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
)

N_HASH_LAYERS = 3
N_LAYERS = 4
NUM_EXPERTS = 4


def _build_model_config():
    return _make_v4_config(
        dim=32,
        n_layers=N_LAYERS,
        vocab_size=512,
        n_heads=4,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=(1,) * N_LAYERS,
        window_size=32,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_inter_dim=16,
        num_experts=NUM_EXPERTS,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=N_HASH_LAYERS,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=32,
        compress_rope_theta=10000.0,
        original_seq_len=32,
        num_mtp_layers=1,
    )


@pytest.fixture(scope="module")
def adapter():
    model_config = _build_model_config()
    return DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)


@pytest.mark.parametrize("layer_id", range(N_HASH_LAYERS))
def test_from_hf_ignores_gate_bias_for_hash_layers(adapter, layer_id):
    hf_state_dict = {
        f"layers.{layer_id}.ffn.gate.bias": torch.arange(
            NUM_EXPERTS, dtype=torch.float32
        )
    }

    assert adapter.from_hf(hf_state_dict) == {}


def test_from_hf_keeps_gate_bias_for_non_hash_layer(adapter):
    hf_state_dict = {
        "layers.3.ffn.gate.bias": torch.arange(NUM_EXPERTS, dtype=torch.float32)
    }

    restored = adapter.from_hf(hf_state_dict)

    assert "layers.3.moe.expert_bias_E" in restored
    torch.testing.assert_close(
        restored["layers.3.moe.expert_bias_E"],
        hf_state_dict["layers.3.ffn.gate.bias"],
        rtol=0,
        atol=0,
    )


def test_from_hf_keeps_gate_bias_for_mtp_layer(adapter):
    hf_state_dict = {
        "mtp.0.ffn.gate.bias": torch.arange(NUM_EXPERTS, dtype=torch.float32)
    }

    restored = adapter.from_hf(hf_state_dict)

    assert "mtp_layers.0.moe.expert_bias_E" in restored
    torch.testing.assert_close(
        restored["mtp_layers.0.moe.expert_bias_E"],
        hf_state_dict["mtp.0.ffn.gate.bias"],
        rtol=0,
        atol=0,
    )
