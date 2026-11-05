# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint conversion tests; no native trainer/runtime compatibility claim."""

import pytest
import torch
from legacy_state_dict import from_legacy_state_dict, to_legacy_state_dict
from model import BlockDiffusionModel, ModelConfig


def config(**changes):
    values = dict(
        vocab_size=13,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        moe_intermediate_size=6,
        num_experts_per_tok=2,
        dtype=torch.bfloat16,
    )
    values.update(changes)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    "num_experts,shared,tied", [(0, 0, False), (0, 0, True), (4, 0, False), (4, 1, False), (4, 1, True)]
)
def test_legacy_roundtrip_preserves_keys_values_dtypes_and_strict_loading(num_experts, shared, tied):
    cfg = config(num_experts=num_experts, num_shared_experts=shared, tie_word_embeddings=tied)
    model = BlockDiffusionModel(cfg)
    source = model.state_dict()
    exported = to_legacy_state_dict(source)
    recovered = from_legacy_state_dict(exported)
    assert recovered.keys() == source.keys()
    for name, expected in source.items():
        assert recovered[name].dtype == expected.dtype
        torch.testing.assert_close(recovered[name], expected, rtol=0, atol=0)
    fresh = BlockDiffusionModel(cfg)
    result = fresh.load_state_dict(recovered, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    if tied:
        assert fresh.lm_head.weight is fresh.embed_tokens.weight


def test_stacked_experts_pack_gate_then_up_and_transpose_down():
    model = BlockDiffusionModel(config(num_experts=4, num_shared_experts=1))
    with torch.no_grad():
        for layer in model.layers:
            for index, expert in enumerate(layer.mlp.experts):
                for offset, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                    weight = getattr(expert, name).weight
                    weight.copy_(
                        (torch.arange(weight.numel()).reshape(weight.shape) + 100 * index + 10 * offset).to(
                            weight.dtype
                        )
                    )
    legacy = to_legacy_state_dict(model.state_dict())
    for layer_index, layer in enumerate(model.layers):
        packed = legacy[f"backbone.layers.{layer_index}.mlp.experts.w13"]
        down = legacy[f"backbone.layers.{layer_index}.mlp.experts.w2"]
        assert packed.shape == (4, 8, 12)
        assert down.shape == (4, 6, 8)
        for index, expert in enumerate(layer.mlp.experts):
            torch.testing.assert_close(packed[index, :, :6], expert.gate_proj.weight.T, rtol=0, atol=0)
            torch.testing.assert_close(packed[index, :, 6:], expert.up_proj.weight.T, rtol=0, atol=0)
            torch.testing.assert_close(down[index], expert.down_proj.weight.T, rtol=0, atol=0)
        assert f"backbone.layers.{layer_index}.mlp.gate.weight" in legacy
        assert f"backbone.layers.{layer_index}.mlp.shared_expert.gate_proj.weight" in legacy


@pytest.mark.parametrize("direction", [to_legacy_state_dict, from_legacy_state_dict])
def test_unknown_checkpoint_keys_are_never_dropped(direction):
    with pytest.raises(ValueError):
        direction({"unsupported.weight": torch.ones(2, 2)})


def test_incomplete_experts_and_noncontiguous_ids_are_rejected():
    model = BlockDiffusionModel(config(num_experts=4))
    state = model.state_dict()
    missing = dict(state)
    del missing["layers.0.mlp.experts.0.up_proj.weight"]
    with pytest.raises(ValueError):
        to_legacy_state_dict(missing)
    noncontiguous = {
        key.replace("layers.0.mlp.experts.3.", "layers.0.mlp.experts.5."): value for key, value in state.items()
    }
    with pytest.raises(ValueError):
        to_legacy_state_dict(noncontiguous)
    legacy = to_legacy_state_dict(state)
    del legacy["backbone.layers.0.mlp.experts.w2"]
    with pytest.raises(ValueError):
        from_legacy_state_dict(legacy)


def test_more_than_one_shared_expert_is_not_silently_collapsed():
    model = BlockDiffusionModel(config(num_experts=4, num_shared_experts=2))
    with pytest.raises(ValueError):
        to_legacy_state_dict(model.state_dict())


def test_incompatible_expert_dtypes_are_rejected_without_promotion():
    model = BlockDiffusionModel(config(num_experts=4))
    state = dict(model.state_dict())
    for name in ("gate_proj", "up_proj", "down_proj"):
        key = f"layers.0.mlp.experts.1.{name}.weight"
        state[key] = state[key].float()
    with pytest.raises(ValueError):
        to_legacy_state_dict(state)
    legacy = to_legacy_state_dict(model.state_dict())
    legacy["backbone.layers.0.mlp.experts.w2"] = legacy["backbone.layers.0.mlp.experts.w2"].float()
    with pytest.raises(ValueError):
        from_legacy_state_dict(legacy)
