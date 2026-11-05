# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent numerical acceptance for the block-diffusion training path."""

import pytest
import torch
import torch.nn.functional as F
from raw_reference import (
    BlockDiffusionModel,
    ModelConfig,
    RoutingPlan,
    TrainingSpec,
    training_attention_mask,
)


def config(**updates):
    fields = dict(
        vocab_size=13,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        num_experts=0,
        num_shared_experts=0,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        dtype=torch.float32,
        kv_dtype=torch.float32,
    )
    fields.update(updates)
    return ModelConfig(**fields)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(912)


def mask_oracle(length, block_size, policy):
    return torch.tensor(
        [
            [
                key // block_size <= query // block_size
                if policy == "block_causal"
                else (key <= query if query < length - block_size else True)
                for key in range(length)
            ]
            for query in range(length)
        ],
        dtype=torch.bool,
    )


@pytest.mark.parametrize("policy", ["block_causal", "prefix_causal_canvas_bidirectional"])
def test_training_mask_matches_independent_visibility_oracle(policy):
    spec = TrainingSpec(block_size=2, attention_policy=policy)
    actual = training_attention_mask(6, spec)
    expected = mask_oracle(6, 2, policy)
    assert actual.dtype == torch.bool
    assert actual.shape == (6, 6)
    torch.testing.assert_close(actual, expected)
    assert actual[-2, -1]  # Canvas queries must see later positions in their block.
    assert not actual[0, -1]


@pytest.mark.parametrize("policy", ["block_causal", "prefix_causal_canvas_bidirectional"])
@torch.no_grad()
def test_future_canvas_cannot_change_prefix_but_canvas_is_bidirectional(policy):
    model = BlockDiffusionModel(config())
    spec = TrainingSpec(block_size=2, attention_policy=policy)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = tokens.clone()
    changed[0, -1] = 9
    labels = tokens.clone()
    original = model.forward_training(tokens, labels, spec=spec).logits
    modified = model.forward_training(changed, labels, spec=spec).logits
    torch.testing.assert_close(original[:, :-2], modified[:, :-2], atol=0, rtol=0)
    assert not torch.allclose(original[:, -2], modified[:, -2], atol=1e-7, rtol=1e-7)


def test_training_loss_uses_only_same_position_nonignored_labels():
    model = BlockDiffusionModel(config())
    spec = TrainingSpec(block_size=2)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[-100, 5, -100, 7], [3, -100, 8, -100]])
    result = model.forward_training(tokens, labels, spec=spec)
    valid = labels != -100
    selected_logits = result.logits[valid]
    selected_targets = labels[valid]
    expected_losses = selected_logits.logsumexp(-1) - selected_logits.gather(1, selected_targets[:, None]).squeeze(1)
    torch.testing.assert_close(result.loss_sum, expected_losses.sum())
    torch.testing.assert_close(result.loss, expected_losses.mean())
    assert result.valid_tokens.ndim == 0
    assert result.valid_tokens.item() == 4
    assert result.logits.shape == (2, 4, 13)
    result.loss.backward()
    # Ignored prompt positions still contribute context and must retain autograd.
    assert model.embed_tokens.weight.grad[1].abs().sum() > 0


def test_empty_loss_targets_produce_differentiable_zero_without_nan():
    model = BlockDiffusionModel(config())
    tokens = torch.tensor([[1, 2, 3, 4]])
    labels = torch.full_like(tokens, -100)
    result = model.forward_training(tokens, labels, spec=TrainingSpec(block_size=2))
    assert result.loss.requires_grad
    assert result.loss.item() == result.loss_sum.item() == 0
    assert result.valid_tokens.item() == 0
    result.loss.backward()
    assert model.embed_tokens.weight.grad is not None
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert parameter.grad.isfinite().all()
            assert torch.count_nonzero(parameter.grad) == 0


def _reference_logits(model, tokens, spec, routing_plan=None):
    """Full training computation using elementary math, with no model forwards."""
    cfg = model.config
    batch, length = tokens.shape
    hidden = F.embedding(tokens, model.embed_tokens.weight)
    mask = mask_oracle(length, spec.block_size, spec.attention_policy)
    frequencies = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
    angles = torch.arange(length).float()[:, None] * frequencies[None]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]

    def norm(x, module):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + cfg.rms_norm_eps) * module.weight

    def rotary(x):
        half = cfg.head_dim // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)

    def expert(x, module):
        return F.linear(
            F.silu(F.linear(x, module.gate_proj.weight)) * F.linear(x, module.up_proj.weight), module.down_proj.weight
        )

    for layer in model.layers:
        x = norm(hidden, layer.input_layernorm)
        attention = layer.self_attn
        q = (
            F.linear(x, attention.q_proj.weight)
            .reshape(batch, length, cfg.num_attention_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        k = (
            F.linear(x, attention.k_proj.weight)
            .reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        v = (
            F.linear(x, attention.v_proj.weight)
            .reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        q, k = rotary(q), rotary(k)
        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        scores = (q @ k.transpose(-1, -2)) / cfg.head_dim**0.5
        scores = scores.masked_fill(~mask, -torch.inf)
        context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, -1)
        hidden = hidden + F.linear(context, attention.o_proj.weight)
        x = norm(hidden, layer.post_attention_layernorm)
        if cfg.num_experts:
            flat = x.reshape(-1, cfg.hidden_size)
            score = F.linear(flat, layer.mlp.router.weight)
            selected = torch.tensor(routing_plan.expert_indices)
            weights = score.gather(1, selected).softmax(-1)
            expert_results = []
            for token, row in enumerate(routing_plan.expert_indices):
                value = torch.zeros_like(flat[token])
                for slot, chosen in enumerate(row):
                    value = value + weights[token, slot] * expert(flat[token], layer.mlp.experts[chosen])
                for shared in layer.mlp.shared_experts:
                    value = value + expert(flat[token], shared)
                expert_results.append(value)
            feedforward = torch.stack(expert_results).reshape_as(x)
        else:
            feedforward = expert(x, layer.mlp)
        hidden = hidden + feedforward
    return F.linear(norm(hidden, model.norm), model.lm_head.weight)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("policy", ["block_causal", "prefix_causal_canvas_bidirectional"])
def test_training_logits_loss_and_all_gradients_match_independent_math(moe, policy):
    cfg = config(num_experts=4 if moe else 0, num_shared_experts=1 if moe else 0)
    model = BlockDiffusionModel(cfg)
    spec = TrainingSpec(block_size=2, attention_policy=policy)
    tokens = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, -100, 4]])
    replay = RoutingPlan(expert_indices=((0, 1),) * 4) if moe else None
    actual = model.forward_training(tokens, labels, spec=spec, routing_plan=replay)
    parameters = tuple(model.parameters())
    actual_grads = torch.autograd.grad(actual.loss, parameters, allow_unused=True)
    expected_logits = _reference_logits(model, tokens, spec, replay)
    valid = labels != spec.ignore_index
    reference_selected = expected_logits[valid]
    expected_loss = (
        reference_selected.logsumexp(-1) - reference_selected.gather(1, labels[valid, None]).squeeze(1)
    ).mean()
    expected_grads = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    torch.testing.assert_close(actual.logits, expected_logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.loss, expected_loss, atol=1e-6, rtol=1e-5)
    for (name, _), result, reference in zip(model.named_parameters(), actual_grads, expected_grads, strict=False):
        assert (result is None) == (reference is None), name
        if result is not None:
            torch.testing.assert_close(result, reference, atol=3e-6, rtol=5e-4, msg=name)


def test_training_gradients_match_central_finite_difference():
    model = BlockDiffusionModel(config(num_hidden_layers=1))
    tokens = torch.tensor([[1, 2]])
    labels = torch.tensor([[-100, 3]])
    spec = TrainingSpec(block_size=2)
    loss = model.forward_training(tokens, labels, spec=spec).loss
    parameter = model.lm_head.weight
    analytic = torch.autograd.grad(loss, parameter)[0][3, 0].item()
    original = parameter[3, 0].item()
    epsilon = 0.001
    with torch.no_grad():
        parameter[3, 0] = original + epsilon
        plus = model.forward_training(tokens, labels, spec=spec).loss.item()
        parameter[3, 0] = original - epsilon
        minus = model.forward_training(tokens, labels, spec=spec).loss.item()
        parameter[3, 0] = original
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), abs=3e-4, rel=3e-3)


def test_training_and_inference_gradient_modes_remain_separate():
    model = BlockDiffusionModel(config())
    tokens = torch.tensor([[1, 2]])
    with torch.enable_grad():
        inference = model(tokens)
        training = model.forward_training(tokens, tokens, spec=TrainingSpec(block_size=2))
    assert not inference.logits.requires_grad
    assert training.logits.requires_grad
    assert training.loss.requires_grad


@pytest.mark.parametrize("moe", [False, True])
def test_training_meta_forward_and_backward_are_value_independent(moe):
    cfg = config(num_experts=4 if moe else 0)
    with torch.device("meta"):
        model = BlockDiffusionModel(cfg)
        tokens = torch.empty((1, 4), dtype=torch.long)
        labels = torch.empty((1, 4), dtype=torch.long)
    replay = RoutingPlan.balanced(4, 4, 2) if moe else None
    result = model.forward_training(tokens, labels, spec=TrainingSpec(block_size=2), routing_plan=replay)
    result.loss.backward()
    assert result.logits.shape == (1, 4, 13)
    assert result.loss.device.type == "meta"
    assert model.embed_tokens.weight.grad.device.type == "meta"


def test_training_meta_make_fx_captures_actual_backward():
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch.device("meta"):
        model = BlockDiffusionModel(config(num_hidden_layers=1))
        tokens = torch.empty((1, 4), dtype=torch.long)
        labels = torch.empty((1, 4), dtype=torch.long)
    parameters = tuple(model.parameters())

    def run(ids, targets):
        result = model.forward_training(ids, targets, spec=TrainingSpec(block_size=2))
        return torch.autograd.grad(result.loss, parameters)

    graph = make_fx(run)(tokens, labels)
    gradients = graph(tokens, labels)
    assert len(gradients) == len(parameters)
    targets = [str(node.target) for node in graph.graph.nodes]
    assert not any("_local_scalar_dense" in name for name in targets)
    assert any("backward" in name for name in targets)
    assert any("mm" in name for name in targets)


@pytest.mark.parametrize("size", [0, -1, True])
def test_training_spec_rejects_invalid_block_size(size):
    with pytest.raises(ValueError):
        TrainingSpec(block_size=size)


@pytest.mark.parametrize("length", [1, 3, 5])
def test_training_mask_rejects_partial_blocks(length):
    with pytest.raises(ValueError):
        training_attention_mask(length, TrainingSpec(block_size=2))
