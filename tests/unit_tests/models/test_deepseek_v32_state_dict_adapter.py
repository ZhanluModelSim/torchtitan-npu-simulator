# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch

from torchtitan_npu.models.deepseek_v32 import deepseekv32_configs
from torchtitan_npu.models.deepseek_v32.state_dict_adapter import (
    DeepSeekV32StateDictAdapter,
)


def _adapter() -> DeepSeekV32StateDictAdapter:
    return DeepSeekV32StateDictAdapter(
        deepseekv32_configs["smoketest"](),
        hf_assets_path=None,
    )


def test_adapter_derives_moe_layout_from_config_tree():
    adapter = _adapter()

    assert adapter.first_k_dense == 1
    assert adapter.n_experts == 8
    assert adapter.use_gmm is True


def test_attention_and_indexer_hf_mapping_round_trip():
    adapter = _adapter()
    hf_state_dict = {
        "model.embed_tokens.weight": torch.randn(3, 4),
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(2, 4),
        "model.layers.0.self_attn.q_a_layernorm.weight": torch.randn(2),
        "model.layers.0.self_attn.q_b_proj.weight": torch.randn(4, 2),
        "model.layers.0.self_attn.indexer.wq_b.weight": torch.randn(2, 2),
        "model.layers.0.self_attn.indexer.wk.weight": torch.randn(2, 4),
        "model.layers.0.self_attn.indexer.k_norm.weight": torch.randn(2),
        "model.layers.0.self_attn.indexer.k_norm.bias": torch.randn(2),
        "model.layers.0.self_attn.indexer.weights_proj.weight": torch.randn(2, 4),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": torch.randn(3, 4),
        "model.layers.0.self_attn.kv_a_layernorm.weight": torch.randn(3),
        "model.layers.0.self_attn.kv_b_proj.weight": torch.randn(4, 3),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4),
        "model.layers.0.input_layernorm.weight": torch.randn(4),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "model.norm.weight": torch.randn(4),
        "lm_head.weight": torch.randn(3, 4),
    }

    titan_state_dict = adapter.from_hf(hf_state_dict)
    assert "layers.0.attention.pre_attention.indexer.wk.weight" in titan_state_dict
    assert "layers.0.attention.post_attention.wo.weight" in titan_state_dict

    rebuilt = adapter.to_hf(titan_state_dict)
    assert rebuilt.keys() == hf_state_dict.keys()
    for key in hf_state_dict:
        torch.testing.assert_close(rebuilt[key], hf_state_dict[key])


def test_grouped_expert_hf_mapping_round_trip():
    adapter = _adapter()
    hf_state_dict = {
        f"model.layers.1.mlp.experts.{expert}.gate_proj.weight": torch.full(
            (2, 2), float(expert)
        )
        for expert in range(adapter.n_experts)
    }

    titan_state_dict = adapter.from_hf(hf_state_dict)
    grouped = titan_state_dict["layers.1.moe.experts.w1"]
    assert grouped.shape == (adapter.n_experts, 2, 2)

    rebuilt = adapter.to_hf(titan_state_dict)
    assert rebuilt.keys() == hf_state_dict.keys()
    for key in hf_state_dict:
        torch.testing.assert_close(rebuilt[key], hf_state_dict[key])


def test_mtp_mapping_is_enabled_by_model_config():
    model_config = deepcopy(deepseekv32_configs["smoketest"]())
    model_config.num_mtp_modules = 1
    adapter = DeepSeekV32StateDictAdapter(model_config, hf_assets_path=None)

    assert adapter.from_hf_map["model.layers.{}.enorm.weight"] == (
        "layers.{}.enorm.weight"
    )
    assert adapter.from_hf_map["model.layers.{}.hnorm.weight"] == (
        "layers.{}.hnorm.weight"
    )
    assert adapter.from_hf_map["model.layers.{}.eh_proj.weight"] == (
        "layers.{}.eh_proj.weight"
    )
