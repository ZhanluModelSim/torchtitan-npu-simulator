# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi HF checkpoint names and grouped expert layouts on the native model."""

import torch

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel


def test_registry_adapter_maps_hf_dense_router_latent_and_kda_norm_weights():
    spec = kimi_k3_debugmodel().model_spec
    adapter = spec.state_dict_adapter(spec.model)
    paths = {
        "language_model.lm_head.weight": ("lm_head.weight", (2048, 256)),
        "language_model.model.layers.0.mlp.gate_proj.weight": ("layers.0.feed_forward.w1.weight", (512, 256)),
        "language_model.model.layers.0.mlp.up_proj.weight": ("layers.0.feed_forward.w3.weight", (512, 256)),
        "language_model.model.layers.0.self_attn.o_norm.weight": ("layers.0.attention.o_norm.norm.weight", (32,)),
        "language_model.model.layers.1.block_sparse_moe.gate.weight": ("layers.1.moe.router.gate.weight", (8, 256)),
        "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias": (
            "layers.1.moe.expert_bias_E",
            (8,),
        ),
        "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight": (
            "layers.1.moe.down_proj.weight",
            (128, 256),
        ),
    }
    source = {hf: torch.full(shape, float(i)) for i, (hf, (_, shape)) in enumerate(paths.items())}
    converted = adapter.from_hf(source)
    assert set(converted) == {target for target, _ in paths.values()}
    for hf, (target, shape) in paths.items():
        assert converted[target] is source[hf]
        assert tuple(converted[target].shape) == shape
    restored = adapter.to_hf(converted)
    assert set(restored) == set(source)
    for key, value in source.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


def test_registry_adapter_preserves_expert_indices_projection_orientation_and_all_model_keys():
    spec = kimi_k3_debugmodel().model_spec
    spec.model.layers[1].moe.routed_experts.inner_experts.hidden_dim = 96
    adapter = spec.state_dict_adapter(spec.model)
    source = {}
    for expert in reversed(range(8)):
        for projection, marker in (("w1", 1), ("w2", 2), ("w3", 3)):
            key = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.{projection}.weight"
            shape = (128, 96) if projection == "w2" else (96, 128)
            source[key] = torch.full(shape, float(expert * 10 + marker))
    converted = adapter.from_hf(source)
    for projection, suffix, marker in (("w1", "EFD", 1), ("w2", "EDF", 2), ("w3", "EFD", 3)):
        key = f"layers.1.moe.routed_experts.inner_experts.{projection}_{suffix}"
        shape = (128, 96) if projection == "w2" else (96, 128)
        assert tuple(converted[key].shape) == (8, *shape)
        for expert in range(8):
            torch.testing.assert_close(
                converted[key][expert], torch.full(shape, float(expert * 10 + marker)), rtol=0, atol=0
            )
    restored = adapter.to_hf(converted)
    for key, value in source.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)

    with torch.device("meta"):
        model = spec.model.build()
    state = model.state_dict()
    assert set(adapter.from_hf(adapter.to_hf(state))) == set(state)


def test_adapter_preserves_low_rank_kda_output_gate_weights():
    spec = kimi_k3_debugmodel().model_spec
    adapter = spec.state_dict_adapter(spec.model)
    source = {
        "language_model.model.layers.0.self_attn.g_a_proj.weight": torch.full((32, 256), 3.0),
        "language_model.model.layers.0.self_attn.g_b_proj.weight": torch.full((256, 32), 7.0),
    }
    state = adapter.from_hf(source)
    assert (
        state["layers.0.attention.g_a_proj.weight"] is source["language_model.model.layers.0.self_attn.g_a_proj.weight"]
    )
    assert (
        state["layers.0.attention.g_b_proj.weight"] is source["language_model.model.layers.0.self_attn.g_b_proj.weight"]
    )
    restored = adapter.to_hf(state)
    for key, value in source.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
