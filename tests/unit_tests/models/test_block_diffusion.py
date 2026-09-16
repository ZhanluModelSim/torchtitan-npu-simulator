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
    from torchtitan.components.loss import cross_entropy_loss
    from torchtitan_npu.models.block_diffusion import model_registry
    from torchtitan_npu.models.block_diffusion.data import corrupt_last_canvas

    config = model_registry(flavor).model
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    tokens = torch.randint(0, config.vocab_size, (1, seq_len))
    corrupted, labels, masked = corrupt_last_canvas(
        tokens[0],
        block_size=config.block_size,
        mask_token_id=config.mask_token_id,
        generator=torch.Generator().manual_seed(0),
        min_mask_ratio=0.5,
        max_mask_ratio=0.5,
    )
    logits = model(corrupted.unsqueeze(0))
    cross_entropy_loss(logits, labels.unsqueeze(0)).backward()

    assert logits.shape == (1, seq_len, config.vocab_size)
    assert masked.sum() == config.block_size // 2
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        _expected_parameter_count(config)
    )
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_attention_uses_one_full_sequence_causal_call(monkeypatch):
    from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA

    captured = []

    def fake_sdpa(q, k, v, **kwargs):
        captured.append((q.shape, k.shape, kwargs["is_causal"]))
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    attention = ScaledCausalSDPA.Config(compute_alpha=0.75).build()
    q = torch.randn(1, 8, 2, 8)
    attention(q, q, q)
    assert captured == [(torch.Size([1, 2, 8, 8]), torch.Size([1, 2, 8, 8]), True)]


def test_npu_attention_converter_emits_one_full_causal_fused_kernel(monkeypatch):
    import torch_npu

    from torchtitan_npu.converters.kernels.block_diffusion_attention import (
        NPUBlockDiffusionAttention,
        NPUBlockDiffusionAttentionConverter,
    )
    from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA

    class AttentionHolder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.attention = ScaledCausalSDPA.Config(compute_alpha=0.75).build()

    captured = []

    def fake_fusion_attention(q, k, v, **kwargs):
        captured.append((q.shape, k.shape, kwargs))
        stats = torch.zeros(
            (q.shape[0], kwargs["head_num"], q.shape[1], 8),
            dtype=torch.float32,
        )
        return torch.zeros_like(q), stats, stats, torch.empty(0), 0, 0, 0

    captured_grads = []

    def fake_fusion_attention_grad(q, k, v, grad_output, **kwargs):
        captured_grads.append((q.shape, k.shape, kwargs))
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v), None, None

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", fake_fusion_attention)
    monkeypatch.setattr(torch_npu, "npu_fusion_attention_grad", fake_fusion_attention_grad)
    model = AttentionHolder()
    NPUBlockDiffusionAttentionConverter(SimpleNamespace(name="block_diffusion")).convert(model)

    q = torch.randn(1, 8, 4, 8, requires_grad=True)
    k = torch.randn(1, 8, 2, 8, requires_grad=True)
    v = torch.randn(1, 8, 2, 8, requires_grad=True)
    output = model.attention(q, k, v, enable_gqa=True)
    output.sum().backward()

    assert isinstance(model.attention, NPUBlockDiffusionAttention)
    assert output.shape == q.shape
    assert [(q_shape, k_shape) for q_shape, k_shape, _ in captured] == [
        (torch.Size([1, 8, 32]), torch.Size([1, 8, 16])),
    ]
    kwargs = captured[0][2]
    assert kwargs["input_layout"] == "BSH"
    assert kwargs["head_num"] == 4
    assert kwargs["sparse_mode"] == 2
    assert kwargs["atten_mask"].shape == (2048, 2048)
    assert kwargs["scale"] == pytest.approx(8**-0.5)
    assert kwargs["pre_tockens"] == 2_147_483_647
    assert kwargs["next_tockens"] == 0
    assert kwargs["inner_precise"] == 0
    assert kwargs["gen_mask_parallel"] is True
    assert kwargs["sync"] is False
    assert len(captured_grads) == 1
    assert captured_grads[0][2]["sparse_mode"] == 2
    assert captured_grads[0][2]["pre_tockens"] == 2_147_483_647
    assert captured_grads[0][2]["next_tockens"] == 0
    assert captured_grads[0][2]["inner_precise"] == 0
    assert all(tensor.grad is not None for tensor in (q, k, v))


def test_simulated_block_diffusion_attention_decomposes_fused_boundaries():
    from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
    from torchtitan_npu.simulator.hardware_shims.block_diffusion_attention import (
        SimBlockDiffusionAttention,
    )

    attention = SimBlockDiffusionAttention(
        SimBlockDiffusionAttention.Config(compute_alpha=0.75)
    )
    q, k, v = [
        torch.empty((1, 8, 2, 8), device="meta", requires_grad=True)
        for _ in range(3)
    ]
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = attention(q, k, v, enable_gqa=True)
        phase["value"] = "backward"
        output.sum().backward()

    nodes = [
        node
        for node in capture.build_nodes().values()
        if node.attrs.get("decomposed_from") == "npu.npu_fusion_attention.default"
    ]
    forward = [node for node in nodes if node.annotations["comp_type"] == "F"]
    backward = [node for node in nodes if node.annotations["comp_type"] == "B"]

    assert output.shape == q.shape
    assert [node.op_type for node in forward] == ["matmul", "softmax", "matmul"]
    assert [node.op_type for node in backward] == ["matmul", "matmul", "softmax", "matmul", "matmul"]
    assert [node.annotations["raw_op_type"] for node in forward] == [
        "aten.matmul.default",
        "aten._softmax.default",
        "aten.matmul.default",
    ]
    assert backward[2].annotations["raw_op_type"] == "aten._softmax_backward_data.default"

    # S=8 causal gives 4 effective keys; alpha=0.75 folds that to 3.
    assert forward[0].attrs["effective_kv_seq_len"] == 3
    assert [meta.shape for meta in forward[0].inputs] == [(2, 8, 8), (2, 8, 3)]
    assert [meta.shape for meta in forward[0].outputs] == [(2, 8, 3)]
    assert [meta.shape for meta in forward[2].inputs] == [(2, 8, 3), (2, 3, 8)]
    # The returned BSH tensor is the final matmul output, preserving the
    # downstream dependency edge while retaining the same element count.
    assert [meta.shape for meta in forward[2].outputs] == [(1, 8, 16)]
    assert forward[2].successors

    assert all(node.attrs["layout"] == "BSH" for node in nodes)
    assert all(node.attrs["compute_alpha"] == pytest.approx(0.75) for node in nodes)
    assert all(node.parameter_inputs["num_heads"] == 2 for node in nodes)
    assert all(node.parameter_inputs["num_kv_heads"] == 2 for node in nodes)
    assert all(node.parameter_inputs["head_dim"] == 8 for node in nodes)
    assert all(node.parameter_inputs["effective_kv_seq_len"] == 3 for node in nodes)
    assert all(tensor.grad is not None for tensor in (q, k, v))


def test_attention_compute_alpha_does_not_change_numerical_forward():
    from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA

    torch.manual_seed(0)
    q = torch.randn(2, 8, 2, 4)
    k = torch.randn(2, 8, 2, 4)
    v = torch.randn(2, 8, 2, 4)
    attention = ScaledCausalSDPA.Config(compute_alpha=0.25).build()
    actual = attention(q, k, v)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_top_level_attention_compute_alpha_override_reaches_built_layers():
    from torchtitan_npu.models.block_diffusion import model_registry
    from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA

    config = model_registry("dense_debug").model
    config.attention_compute_alpha = 0.5
    model = config.build()
    alphas = [module.compute_alpha for module in model.modules() if isinstance(module, ScaledCausalSDPA)]

    assert alphas == [0.5] * config.n_layers


def test_attention_compute_alpha_must_be_positive():
    from torchtitan_npu.models.block_diffusion import model_registry

    config = model_registry("dense_debug").model
    config.attention_compute_alpha = 0.0
    with pytest.raises(ValueError, match="attention_compute_alpha"):
        config.__post_init__()


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


def test_dataloader_slides_by_one_canvas_and_restores_corruption_rng():
    from torchtitan.components.loss import IGNORE_INDEX
    from torchtitan_npu.models.block_diffusion.config_registry import (
        block_diffusion_smoketest,
    )

    config = block_diffusion_smoketest()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

    def build_loader():
        return config.dataloader.build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=1,
        )

    loader = build_loader()
    iterator = iter(loader)
    first_inputs, first_labels = next(iterator)
    checkpoint = loader.state_dict()
    second_inputs, second_labels = next(iterator)

    restored_loader = build_loader()
    restored_loader.load_state_dict(checkpoint)
    restored_inputs, restored_labels = next(iter(restored_loader))
    assert torch.equal(second_inputs["input"], restored_inputs["input"])
    assert torch.equal(second_labels, restored_labels)

    def reconstruct(inputs, labels):
        return torch.where(labels != IGNORE_INDEX, labels, inputs["input"])

    block_size = config.dataloader.block_size
    first_clean = reconstruct(first_inputs, first_labels)
    second_clean = reconstruct(second_inputs, second_labels)
    assert torch.equal(first_clean[:, block_size:], second_clean[:, :-block_size])


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
        "attention_compute_alpha",
    ):
        assert getattr(simulation.model_spec.model, field) == getattr(
            training.model_spec.model, field
        )
    assert simulation.parallelism == training.parallelism
    assert simulation.training == training.training
    assert type(simulation.dataloader) is type(training.dataloader)
    assert type(training.validator.dataloader) is type(training.dataloader)
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
