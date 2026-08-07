# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.models.deepseek_v32 import _make_dsv32_model_config
from torchtitan_npu.models.deepseek_v32.state_dict_adapter import (
    DeepSeekV32StateDictAdapter,
)


def _model_config(*, num_mtp_modules: int):
    return _make_dsv32_model_config(
        vocab_size=32,
        dim=16,
        inter_dim=32,
        moe_inter_dim=8,
        n_layers=2,
        n_dense_layers=1,
        n_heads=2,
        num_experts=8,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        num_mtp_modules=num_mtp_modules,
    )


def test_to_hf_exports_shared_weights_for_each_mtp_layer():
    model_config = _model_config(num_mtp_modules=2)
    adapter = DeepSeekV32StateDictAdapter(model_config)
    state_dict = {
        "tok_embeddings.weight": torch.randn(32, 16),
        "norm.weight": torch.randn(16),
        "output.weight": torch.randn(32, 16),
    }

    hf_state_dict = adapter.to_hf(state_dict)

    for layer_id in (2, 3):
        assert (
            hf_state_dict[f"model.layers.{layer_id}.embed_tokens.weight"] is hf_state_dict["model.embed_tokens.weight"]
        )
        assert hf_state_dict[f"model.layers.{layer_id}.shared_head.norm.weight"] is hf_state_dict["model.norm.weight"]
        assert hf_state_dict[f"model.layers.{layer_id}.shared_head.head.weight"] is hf_state_dict["lm_head.weight"]


def test_to_hf_exports_available_mtp_shared_weights_from_partial_state_dict():
    adapter = DeepSeekV32StateDictAdapter(_model_config(num_mtp_modules=1))
    embedding = torch.randn(32, 16)

    hf_state_dict = adapter.to_hf({"tok_embeddings.weight": embedding})

    assert hf_state_dict["model.layers.2.embed_tokens.weight"] is embedding
    assert not any("shared_head" in key for key in hf_state_dict)


def test_mtp_shared_weight_aliases_round_trip_to_main_model_keys():
    adapter = DeepSeekV32StateDictAdapter(_model_config(num_mtp_modules=1))
    state_dict = {
        "tok_embeddings.weight": torch.randn(32, 16),
        "norm.weight": torch.randn(16),
        "output.weight": torch.randn(32, 16),
    }

    restored_state_dict = adapter.from_hf(adapter.to_hf(state_dict))

    assert restored_state_dict.keys() == state_dict.keys()
    for key, value in state_dict.items():
        assert restored_state_dict[key] is value


def test_to_hf_does_not_export_mtp_shared_weights_when_mtp_is_disabled():
    adapter = DeepSeekV32StateDictAdapter(_model_config(num_mtp_modules=0))

    hf_state_dict = adapter.to_hf(
        {
            "tok_embeddings.weight": torch.randn(32, 16),
            "norm.weight": torch.randn(16),
            "output.weight": torch.randn(32, 16),
        }
    )

    assert not any("shared_head" in key for key in hf_state_dict)
