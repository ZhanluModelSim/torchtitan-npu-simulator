# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit coverage for the native Block Diffusion model."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch


def _expected_parameter_count(config) -> int:
    dim = config.dim
    attention = 2 * dim * config.head_dim * (config.n_heads + config.n_kv_heads)
    norms = 2 * dim
    if config.num_experts:
        ffn = (
            config.num_experts * dim
            + 3 * config.num_experts * dim * config.moe_intermediate_size
            + 3 * dim * config.intermediate_size
        )
    else:
        ffn = 3 * dim * config.intermediate_size
    embedding_and_output = config.vocab_size * dim * (
        1 if config.enable_weight_tying else 2
    )
    return (
        embedding_and_output
        + dim
        + config.n_layers * (attention + norms + ffn)
    )


@pytest.mark.parametrize("flavor", ("dense_debug", "debug", "reduced", "full"))
def test_model_registry_exposes_all_flavors(flavor):
    from torchtitan_npu.models.block_diffusion import model_registry

    spec = model_registry(flavor)
    assert spec.name == "block_diffusion"
    assert spec.flavor == flavor


def test_full_flavor_matches_10_05t_parameter_contract():
    from torchtitan_npu.models.block_diffusion import model_registry

    config = model_registry("full").model
    assert _expected_parameter_count(config) == 10_052_615_823_360


def test_model_registry_rejects_unknown_flavor():
    from torchtitan_npu.models.block_diffusion import model_registry

    with pytest.raises(ValueError, match="Unknown Block Diffusion flavor"):
        model_registry("missing")


@pytest.mark.parametrize(
    ("flavor", "seq_len"),
    (("dense_debug", 16), ("debug", 32)),
)
def test_prefix_canvas_forward_backward_and_parameter_formula(flavor, seq_len):
    from torchtitan_npu.models.block_diffusion import model_registry

    config = model_registry(flavor).model
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    tokens = torch.randint(0, config.vocab_size, (1, seq_len))
    logits = model(tokens)
    logits.float().mean().backward()

    assert logits.shape == (1, seq_len, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        _expected_parameter_count(config)
    )
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_attention_splits_causal_prefix_and_bidirectional_canvas(monkeypatch):
    from torchtitan_npu.models.block_diffusion.attention import PrefixCanvasSDPA

    captured = []

    def fake_sdpa(q, k, v, **kwargs):
        captured.append((q.shape, k.shape, kwargs["is_causal"]))
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    attention = PrefixCanvasSDPA.Config(block_size=4).build()
    q = torch.randn(1, 8, 2, 8)
    attention(q, q, q)
    assert captured == [
        (torch.Size([1, 2, 4, 8]), torch.Size([1, 2, 4, 8]), True),
        (torch.Size([1, 2, 4, 8]), torch.Size([1, 2, 8, 8]), False),
    ]


def test_prefix_canvas_attention_matches_explicit_reference_mask():
    from torchtitan_npu.models.block_diffusion.attention import (
        build_prefix_canvas_attention_mask,
        PrefixCanvasSDPA,
    )

    torch.manual_seed(0)
    q = torch.randn(2, 8, 2, 4)
    k = torch.randn(2, 8, 2, 4)
    v = torch.randn(2, 8, 2, 4)
    attention = PrefixCanvasSDPA.Config(block_size=4).build()
    actual = attention(q, k, v)

    visible = build_prefix_canvas_attention_mask(8, 4)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=visible,
        is_causal=False,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)

    assert visible[:4].equal(torch.ones(4, 8, dtype=torch.bool).tril())
    assert visible[4:].all()


def test_corruption_only_targets_masked_positions_in_final_canvas():
    from torchtitan.components.loss import IGNORE_INDEX
    from torchtitan_npu.models.block_diffusion.data import corrupt_last_canvas

    tokens = torch.arange(32)
    corrupted, labels, masked = corrupt_last_canvas(
        tokens,
        block_size=8,
        mask_token_id=100,
        generator=torch.Generator().manual_seed(7),
        min_mask_ratio=0.25,
        max_mask_ratio=0.25,
    )

    assert masked[:24].sum() == 0
    assert masked[24:].sum() == 2
    assert torch.equal(corrupted[:24], tokens[:24])
    assert (corrupted[masked] == 100).all()
    assert (labels[~masked] == IGNORE_INDEX).all()
    assert torch.equal(labels[masked], tokens[masked])


def test_npu_moe_gate_score_is_meta_safe():
    from torchtitan_npu.converters.kernels.moe_dispatch import _local_gate_scores

    with torch.device("meta"):
        gate = torch.nn.Linear(8, 4, bias=False)
        inputs = torch.empty(3, 8)
    scores = _local_gate_scores(SimpleNamespace(gate=gate), inputs)
    assert scores.device.type == "meta"
    assert scores.shape == (3, 4)


def test_config_rejects_invalid_architecture():
    from torchtitan_npu.models.block_diffusion import model_registry

    config = model_registry("debug").model
    with pytest.raises(ValueError, match="n_kv_heads must divide n_heads"):
        dataclasses.replace(config, n_kv_heads=3)
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        dataclasses.replace(config, num_experts_per_tok=9)


def test_runtime_config_requires_block_aligned_sequence_and_valid_parallel_degrees():
    from torchtitan_npu.models.block_diffusion import model_registry

    config = model_registry("debug").model
    valid_parallelism = SimpleNamespace(
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
    )
    config.update_from_config(
        trainer_config=SimpleNamespace(
            training=SimpleNamespace(seq_len=config.block_size * 4),
            parallelism=valid_parallelism,
        )
    )
    with pytest.raises(ValueError, match="positive multiple"):
        config.update_from_config(
            trainer_config=SimpleNamespace(
                training=SimpleNamespace(seq_len=config.block_size + 1),
                parallelism=valid_parallelism,
            )
        )
    with pytest.raises(ValueError, match="must divide both n_heads"):
        config.update_from_config(
            trainer_config=SimpleNamespace(
                training=SimpleNamespace(seq_len=config.block_size),
                parallelism=SimpleNamespace(
                    tensor_parallel_degree=3,
                    context_parallel_degree=1,
                    expert_parallel_degree=1,
                    expert_tensor_parallel_degree=1,
                ),
            )
        )
    with pytest.raises(ValueError, match="simultaneous expert_parallel_degree"):
        config.update_from_config(
            trainer_config=SimpleNamespace(
                training=SimpleNamespace(seq_len=config.block_size),
                parallelism=SimpleNamespace(
                    tensor_parallel_degree=2,
                    context_parallel_degree=1,
                    expert_parallel_degree=2,
                    expert_tensor_parallel_degree=2,
                ),
            )
        )


def test_raw_state_dict_round_trip_preserves_expert_layout():
    from torchtitan_npu.models.block_diffusion import model_registry
    from torchtitan_npu.models.block_diffusion.state_dict_adapter import (
        BlockDiffusionStateDictAdapter,
    )

    config = model_registry("debug").model
    adapter = BlockDiffusionStateDictAdapter(config)
    experts = config.num_experts
    dim = config.dim
    hidden = config.moe_intermediate_size
    raw = {
        "backbone.embed_tokens.weight": torch.randn(config.vocab_size, dim),
        "backbone.layers.0.input_layernorm.weight": torch.randn(dim),
        "backbone.layers.0.mlp.experts.w13": torch.randn(
            experts, dim, 2 * hidden
        ),
        "backbone.layers.0.mlp.experts.w2": torch.randn(experts, hidden, dim),
        "lm_head.weight": torch.randn(config.vocab_size, dim),
    }

    restored = adapter.to_hf(adapter.from_hf(raw))
    assert restored.keys() == raw.keys()
    for key in raw:
        torch.testing.assert_close(restored[key], raw[key])


def test_training_and_simulator_configs_share_model_and_parallelism():
    from torchtitan_npu.models.block_diffusion.config_registry import (
        block_diffusion_smoketest as training_config,
    )
    from torchtitan_npu.simulator.config_registry import (
        block_diffusion_smoketest as simulator_config,
    )

    training = training_config()
    simulation = simulator_config()
    assert simulation.model_spec.name == training.model_spec.name
    assert simulation.model_spec.flavor == training.model_spec.flavor
    assert type(simulation.model_spec.model) is type(training.model_spec.model)
    for field in (
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "num_experts",
        "block_size",
    ):
        assert getattr(simulation.model_spec.model, field) == getattr(
            training.model_spec.model, field
        )
    assert simulation.parallelism == training.parallelism
    assert simulation.training == training.training
    assert simulation.compile.enable is False
    assert simulation.simulation.output_dir.endswith("block_diffusion_smoketest")


def test_full_recipe_has_a_valid_expert_domain():
    from torchtitan_npu.models.block_diffusion.config_registry import (
        block_diffusion_baseline,
    )

    config = block_diffusion_baseline()
    parallelism = config.parallelism
    assert config.training.seq_len == 4096
    assert config.dataloader.block_size == 256
    assert config.dataloader.mask_token_id == config.model_spec.model.mask_token_id
    expert_domain = (
        parallelism.data_parallel_shard_degree
        * parallelism.context_parallel_degree
        * parallelism.tensor_parallel_degree
    )
    assert expert_domain == 512
    assert expert_domain % parallelism.expert_parallel_degree == 0
