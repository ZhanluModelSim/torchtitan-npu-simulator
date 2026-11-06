# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent equation-based ALoDLM checks; no upstream implementation is copied."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from raw_reference import BlockDiffusionModel, ModelConfig, load_raw_reference
from recipes import alodlm as raw

canonical = load_raw_reference()


def config(**changes):
    values = dict(
        vocab_size=11,
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        num_experts=0,
        dtype=torch.float32,
    )
    values.update(changes)
    return ModelConfig(**values)


def spec(**changes):
    values = dict(
        loop_start=1,
        loop_end=2,
        max_depth=3,
        mask_token_id=0,
        feedback_policy="cumulative",
        normalize_carry=True,
        qk_norm=True,
    )
    values.update(changes)
    return raw.ALoDLMSpec(**values)


def make_batch(two_rows=False):
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]] if two_rows else [[1, 2]])
    labels = ids.clone()
    masked = torch.tensor([[True, False, True, False], [False, True, False, False]] if two_rows else [[True, True]])
    probability = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.25, 0.25]] if two_rows else [[0.5, 0.5]])
    return raw.build_alodlm_batch(ids, labels, masked, probability, block_size=2, mask_token_id=0)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(171)


def test_alodlm_contract_is_separate_from_native_standard_contract():
    assert canonical.MODEL_CONTRACT_VERSION == 3
    assert canonical.STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION == 1
    assert canonical.LOOP_CONTRACT_VERSION == 1
    assert raw.ALODLM_CONTRACT_VERSION == 1
    assert not any("alodlm" in name.lower() for name in vars(canonical))


def test_dual_stream_batch_reorders_observed_first_without_changing_logical_positions():
    batch = make_batch(two_rows=True)
    assert batch.input_ids.shape == (2, 8)
    assert batch.source_length == 4
    assert batch.positions.tolist() == [[0, 1, 2, 3, 1, 0, 3, 2], [0, 1, 2, 3, 0, 1, 2, 3]]
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 2, 0, 4, 0], [4, 3, 2, 1, 4, 0, 2, 1]]
    assert batch.masked.tolist() == [
        [False, False, False, False, False, True, False, True],
        [False, False, False, False, False, True, False, False],
    ]
    assert batch.targets.tolist() == [[1, 2, 3, 4, 2, 1, 4, 3], [4, 3, 2, 1, 4, 3, 2, 1]]
    assert torch.equal(batch.ar_labels[:, :4], torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
    assert (batch.ar_labels[:, 4:] == -100).all()
    assert batch.mask_probability[0, 5].item() == 0.25
    assert batch.mask_probability[0, 7].item() == 0.5
    assert batch.mask_probability[1, 5].item() == 0.75
    expected = torch.zeros((8, 8), dtype=torch.bool)
    for query in range(8):
        query_stream, query_local = divmod(query, 4)
        for key in range(8):
            key_stream, key_local = divmod(key, 4)
            earlier_clean_block = key_stream == 0 and key_local // 2 < query_local // 2
            causal_same_stream_block = (
                query_stream == key_stream and query_local // 2 == key_local // 2 and key_local <= query_local
            )
            expected[query, key] = earlier_clean_block or causal_same_stream_block
    torch.testing.assert_close(batch.attention_mask, expected)
    assert not expected[5, 0]  # No access to the current clean block's target.
    assert expected[7, 0]  # Earlier clean blocks are legitimate context.
    assert not expected[0, 4]


def _rms(tensor, weight, epsilon):
    normalized = tensor.float() * torch.rsqrt(tensor.float().square().mean(-1, keepdim=True) + epsilon)
    return normalized.to(tensor.dtype) * weight


def _block_math(layer, hidden, positions, visible, cfg):
    """Dense transformer equations independent of production forward helpers."""
    normed = _rms(hidden, layer.input_layernorm.weight, cfg.rms_norm_eps)
    batch, length, _ = hidden.shape
    attention = layer.self_attn
    query = (
        F.linear(normed, attention.q_proj.weight)
        .reshape(batch, length, cfg.num_attention_heads, cfg.head_dim)
        .transpose(1, 2)
    )
    key = (
        F.linear(normed, attention.k_proj.weight)
        .reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)
        .transpose(1, 2)
    )
    value = (
        F.linear(normed, attention.v_proj.weight)
        .reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)
        .transpose(1, 2)
    )
    if hasattr(attention.q_norm, "weight"):
        query = _rms(query, attention.q_norm.weight, cfg.rms_norm_eps)
        key = _rms(key, attention.k_norm.weight, cfg.rms_norm_eps)
    frequency = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
    angles = positions.float()[..., None] * frequency[None, None, :]
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]

    def rotate(tensor):
        first, second = tensor.chunk(2, -1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)

    query, key = rotate(query), rotate(key)
    repeats = cfg.num_attention_heads // cfg.num_key_value_heads
    key, value = key.repeat_interleave(repeats, 1), value.repeat_interleave(repeats, 1)
    scores = query @ key.transpose(-1, -2) / cfg.head_dim**0.5
    context = scores.masked_fill(~visible, -torch.inf).softmax(-1) @ value
    context = context.transpose(1, 2).reshape(batch, length, -1)
    hidden = hidden + F.linear(context, attention.o_proj.weight)
    normed = _rms(hidden, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
    feedforward = F.silu(F.linear(normed, layer.mlp.gate_proj.weight)) * F.linear(normed, layer.mlp.up_proj.weight)
    return hidden + F.linear(feedforward, layer.mlp.down_proj.weight)


def _three_pass_reference(model, batch, exits):
    cfg, backbone = model.backbone.config, model.backbone
    apply = lambda layer, hidden: _block_math(layer, hidden, batch.positions, batch.attention_mask, cfg)
    normalize = lambda hidden: _rms(hidden, backbone.norm.weight, cfg.rms_norm_eps)
    embedded = F.embedding(batch.input_ids, backbone.embed_tokens.weight)
    teacher = F.embedding(batch.targets, backbone.embed_tokens.weight)
    initial = apply(backbone.layers[0], embedded)
    first = apply(backbone.layers[1], initial)
    readout1 = normalize(apply(backbone.layers[2], first))
    second_input = torch.where((batch.masked & (exits <= 1))[..., None], teacher, normalize(first))
    second = apply(backbone.layers[1], second_input)
    readout2 = normalize(apply(backbone.layers[2], second))
    third_input = torch.where((batch.masked & (exits <= 2))[..., None], teacher, normalize(second))
    third = apply(backbone.layers[1], third_input)
    readout3 = normalize(apply(backbone.layers[2], third))
    logits = torch.stack([F.linear(readout, backbone.lm_head.weight) for readout in (readout1, readout2, readout3)])
    return logits, third


@pytest.mark.parametrize("two_rows", [False, True])
def test_shared_recurrence_logits_carry_and_gradients_match_hand_unrolled_equations(two_rows):
    model = raw.ALoDLMModel(config(), spec())
    batch = make_batch(two_rows=two_rows)
    exits = torch.where(batch.masked, 3, 0)
    for index, (row, position) in enumerate(batch.masked.nonzero().tolist()):
        exits[row, position] = (1, 3, 2)[index]
    result = model.rollout_training(batch, exit_depths=exits)
    expected, carry = _three_pass_reference(model, batch, exits)
    torch.testing.assert_close(result.logits, expected, atol=5e-6, rtol=4e-5)
    torch.testing.assert_close(result.final_carry, carry, atol=5e-6, rtol=4e-5)
    assert result.logits.shape == (3, *batch.input_ids.shape, 11)
    assert result.gate_logits.shape == (3, *batch.input_ids.shape)
    parameters = tuple(model.backbone.parameters())
    actual_gradients = torch.autograd.grad(result.logits.square().mean(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
    for (name, _), actual, reference in zip(
        model.backbone.named_parameters(), actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=5e-4, msg=name)


def test_training_runs_all_depths_and_keeps_committed_rows_in_dense_work():
    model = raw.ALoDLMModel(config(), spec(max_depth=4))
    batch = make_batch()
    calls = [[] for _ in model.backbone.layers]
    handles = [
        layer.register_forward_pre_hook(lambda _module, args, index=index: calls[index].append(tuple(args[0].shape)))
        for index, layer in enumerate(model.backbone.layers)
    ]
    model.rollout_training(batch, exit_depths=((0, 0, 1, 1),))
    for handle in handles:
        handle.remove()
    assert [len(entries) for entries in calls] == [1, 4, 4]
    assert all(shape == (1, 4, 8) for entries in calls for shape in entries)
    assert model.layer_evaluations() == 9
    assert model.logical_buffer_count() == 6
    other = raw.ALoDLMModel(config(), spec(max_depth=2))
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        - sum(parameter.numel() for parameter in other.parameters())
        == 2
    )


def test_gold_feedback_is_after_exit_only_and_observed_targets_never_reinjected():
    model = raw.ALoDLMModel(config(), spec())
    batch = make_batch()
    exits = ((0, 0, 1, 3),)
    original = model.rollout_training(batch, exit_depths=exits)
    altered_targets = batch.targets.clone()
    altered_targets[0, 2] = 7
    changed = model.rollout_training(replace(batch, targets=altered_targets), exit_depths=exits)
    torch.testing.assert_close(original.logits[0], changed.logits[0], atol=0, rtol=0)
    assert not torch.allclose(original.logits[1, :, 2:], changed.logits[1, :, 2:])
    observed_targets = batch.targets.clone()
    observed_targets[:, :2] = 9
    observed = model.rollout_training(replace(batch, targets=observed_targets), exit_depths=exits)
    torch.testing.assert_close(original.logits, observed.logits, atol=0, rtol=0)


def test_gate_features_are_detached_even_with_nonzero_projection_weights():
    model = raw.ALoDLMModel(config(), spec())
    with torch.no_grad():
        model.exit_gate.projection.weight.fill_(0.2)
    output = model.rollout_training(make_batch(), exit_depths=((0, 0, 2, 3),))
    output.gate_logits.sum().backward()
    assert model.exit_gate.projection.weight.grad is not None
    assert model.exit_gate.projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_inference_feedback_profiles_differ_only_after_a_prior_commit():
    cumulative = raw.ALoDLMModel(config(), spec(feedback_policy="cumulative"))
    newly = raw.ALoDLMModel(config(), spec(feedback_policy="newly_committed"))
    newly.load_state_dict(cumulative.state_dict(), strict=True)
    inputs = torch.tensor([[4, 0, 0]])
    positions = torch.tensor([[0, 1, 2]])
    visible = torch.ones((3, 3), dtype=torch.bool).tril()
    masked = torch.tensor([[False, True, True]])
    kwargs = dict(commit_depths=((0, 1, 3),), feedback_tokens=torch.tensor([[4, 5, 6]]))
    first = cumulative.run_outer(inputs, positions, visible, masked, **kwargs)
    second = newly.run_outer(inputs, positions, visible, masked, **kwargs)
    torch.testing.assert_close(first.logits[:2], second.logits[:2], atol=0, rtol=0)
    assert not torch.allclose(first.logits[2], second.logits[2])
    assert first.tokens.tolist() == [[4, 5, 6]]
    assert first.executed_steps == 3
    restarted = cumulative.run_outer(inputs, positions, visible, masked, **kwargs)
    torch.testing.assert_close(first.logits, restarted.logits, atol=0, rtol=0)


def test_uncached_profile_rejects_standard_kv_and_moe_architecture():
    model = raw.ALoDLMModel(config(), spec())
    batch = make_batch()
    ordinary = BlockDiffusionModel(config())(torch.tensor([[1, 2]]), phase="prefill").cache
    with pytest.raises((ValueError, NotImplementedError)):
        model.rollout_training(batch, exit_depths=((0, 0, 1, 3),), cache=ordinary)
    with pytest.raises(ValueError):
        raw.ALoDLMModel(config(num_experts=4), spec())


def test_alodlm_meta_replay_and_graph_capture_do_not_read_tensor_values():
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch.device("meta"):
        model = raw.ALoDLMModel(config(), spec())
        ids = torch.empty((1, 2), dtype=torch.long)
        batch = raw.build_alodlm_batch(
            ids,
            ids,
            torch.empty_like(ids, dtype=torch.bool),
            torch.empty_like(ids, dtype=torch.float32),
            block_size=2,
            mask_token_id=0,
        )
    graph = make_fx(
        lambda values: model.rollout_training(replace(batch, input_ids=values), exit_depths=((0, 0, 1, 3),)).logits
    )(batch.input_ids)
    assert graph(batch.input_ids).shape == (3, 1, 4, 11)
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)


def _synthetic_objective_inputs():
    batch = make_batch(two_rows=True)
    logits = torch.randn((3, 2, 8, 7), requires_grad=True)
    gate = (torch.randn((3, 2, 8)) * 0.7).requires_grad_()
    depths = torch.full((2, 8), 3, dtype=torch.long)
    depths[0, 5], depths[0, 7], depths[1, 5] = 1, 3, 2
    output = raw.ALoDLMOutput(logits, gate, depths, torch.zeros((2, 8, 8)))
    return output, batch


def _objective_oracle(output, batch, prior_c, individual, aggregate_weight):
    """Direct categorical-product and sequence-credit equations, not release code."""
    depth_count = output.logits.shape[0]
    entries = []
    for batch_index in range(batch.masked.shape[0]):
        for position in range(batch.masked.shape[1]):
            if not batch.masked[batch_index, position]:
                continue
            survival = output.gate_logits.new_ones(())
            probabilities, survival_before = [], []
            for depth in range(depth_count):
                survival_before.append(survival)
                hazard = (
                    output.gate_logits[depth, batch_index, position].sigmoid()
                    if depth < depth_count - 1
                    else survival.new_ones(())
                )
                probabilities.append(survival * hazard)
                survival = survival * (1 - hazard)
            distribution = torch.stack(probabilities)
            exit_depth = int(output.exit_depths[batch_index, position])
            supervision = torch.stack(
                [
                    probabilities[depth]
                    if depth < exit_depth - 1
                    else survival_before[depth]
                    if depth == exit_depth - 1
                    else survival.new_zeros(())
                    for depth in range(depth_count)
                ]
            ).detach()
            row_logits = output.logits[:, batch_index, position]
            ce = row_logits.logsumexp(-1) - row_logits[:, batch.targets[batch_index, position]]
            inverse = 1 / (batch.mask_probability[batch_index, position] + 1e-8)
            entries.append((batch_index, exit_depth - 1, distribution, supervision, ce, inverse))
    zero = output.logits.sum() * 0 + output.gate_logits.sum() * 0
    denoising, actor = zero, zero
    if entries:
        inverse_total = torch.stack([entry[5] for entry in entries]).sum()
        denoising = sum(
            (weights * ce).sum() * inverse / inverse_total for _batch, _exit, _q, weights, ce, inverse in entries
        )
        mean_distribution = torch.stack([entry[2].detach() for entry in entries]).mean(0)
        prior = torch.softmax(-prior_c * torch.arange(1, depth_count + 1).float(), 0)
        for sequence in range(batch.input_ids.shape[0]):
            sequence_credit, sequence_log_probability = zero.detach(), zero
            for batch_index, exit_index, distribution, _weights, ce, _inverse in entries:
                if batch_index != sequence:
                    continue
                log_probability = distribution[exit_index].log()
                baseline_adjusted_cost = (ce[exit_index] - ce[0]).detach()
                marginal = mean_distribution[exit_index].clamp_min(1e-12).log()
                credit = (
                    baseline_adjusted_cost
                    + individual * (log_probability.detach() - marginal)
                    + aggregate_weight * (marginal - prior[exit_index].log())
                )
                sequence_credit = sequence_credit + credit
                sequence_log_probability = sequence_log_probability + log_probability
            actor = actor + sequence_credit.detach() * sequence_log_probability / len(entries)
    ar_terms = []
    for depth in range(depth_count):
        current = []
        for batch_index in range(batch.input_ids.shape[0]):
            for position in range(batch.source_length - 1):
                target = int(batch.ar_labels[batch_index, position + 1])
                if target != -100:
                    scores = output.logits[depth, batch_index, position]
                    current.append(scores.logsumexp(0) - scores[target])
        ar_terms.append(torch.stack(current).mean() if current else zero)
    autoregressive = torch.stack(ar_terms).mean()
    return denoising, actor, autoregressive, (denoising + actor + autoregressive) / 2


def test_practical_objective_values_and_gradients_match_sequence_equation_oracle():
    output, batch = _synthetic_objective_inputs()
    settings = dict(depth_prior_c=0.6, kl_individual=0.2, kl_aggregate=0.7)
    actual = raw.alodlm_training_objective(output, batch, **settings)
    expected = _objective_oracle(output, batch, 0.6, 0.2, 0.7)
    for result, reference in zip(
        (actual.denoising, actual.actor, actual.autoregressive, actual.loss), expected, strict=True
    ):
        torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-5)
    actual_gradients = torch.autograd.grad(actual.loss, (output.logits, output.gate_logits), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected[-1], (output.logits, output.gate_logits), retain_graph=True)
    for result, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, reference, atol=3e-6, rtol=4e-5)
    assert actual.masked_tokens.item() == 3
    assert not actual.denoising_weights.requires_grad
    torch.testing.assert_close(actual.exit_probabilities.sum(0), batch.masked.float())
    torch.testing.assert_close(actual.denoising_weights.sum(0), batch.masked.float())
    assert torch.count_nonzero(actual_gradients[1][-1]) == 0  # Forced terminal hazard.


def test_denoiser_and_actor_gradient_paths_are_separated():
    output, batch = _synthetic_objective_inputs()
    result = raw.alodlm_training_objective(output, batch)
    gate_from_denoiser = torch.autograd.grad(
        result.denoising, output.gate_logits, allow_unused=True, retain_graph=True
    )[0]
    logits_from_actor = torch.autograd.grad(result.actor, output.logits, allow_unused=True, retain_graph=True)[0]
    assert gate_from_denoiser is None or torch.count_nonzero(gate_from_denoiser) == 0
    assert logits_from_actor is None or torch.count_nonzero(logits_from_actor) == 0
    denoiser_gradient = torch.autograd.grad(result.denoising, output.logits)[0]
    assert torch.count_nonzero(denoiser_gradient[1:, 0, 5]) == 0  # Token exited at depth one.
    assert torch.count_nonzero(denoiser_gradient[:, ~batch.masked]) == 0


def test_exit_initialization_and_terminal_profile_are_forced_correctly():
    model = raw.ALoDLMModel(config(), spec(gate_init_mass=(0.2, 0.3, 0.5)))
    batch = make_batch()
    output = model.rollout_training(batch, exit_depths=((0, 0, 1, 3),))
    result = raw.alodlm_training_objective(output, batch)
    torch.testing.assert_close(result.exit_probabilities[:, 0, 2], torch.tensor([0.2, 0.3, 0.5]))
    gates = output.gate_logits.clone()
    gates[-1] = 1000
    changed = raw.alodlm_training_objective(replace(output, gate_logits=gates), batch)
    torch.testing.assert_close(changed.exit_probabilities, result.exit_probabilities)
    torch.testing.assert_close(changed.loss, result.loss)


@pytest.mark.parametrize("supervised_ar", [False, True])
def test_zero_masked_targets_keep_ar_depth_average_and_differentiable_zero(supervised_ar):
    output, batch = _synthetic_objective_inputs()
    empty = replace(
        batch, masked=torch.zeros_like(batch.masked), mask_probability=torch.zeros_like(batch.mask_probability)
    )
    if not supervised_ar:
        empty = replace(empty, ar_labels=torch.full_like(empty.ar_labels, -100))
    result = raw.alodlm_training_objective(output, empty)
    expected = _objective_oracle(output, empty, 0.4, 0.1, 1.0)
    assert result.masked_tokens.item() == 0
    assert result.denoising.item() == result.actor.item() == 0
    torch.testing.assert_close(result.autoregressive, expected[2])
    torch.testing.assert_close(result.loss, result.autoregressive / 2)
    result.loss.backward()
    assert output.logits.grad is not None and output.gate_logits.grad is not None
    assert output.logits.grad.isfinite().all() and output.gate_logits.grad.isfinite().all()
    assert torch.count_nonzero(output.gate_logits.grad) == 0
    if not supervised_ar:
        assert torch.count_nonzero(output.logits.grad) == 0


def test_gate_halt_and_entropy_commit_are_independent_decisions():
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    active = torch.ones((1, 2), dtype=torch.bool)
    absent = torch.zeros_like(active)
    result = raw.alodlm_decision(
        logits, torch.full((1, 2), 4.0), active, absent, torch.zeros((1, 2)), entropy_threshold=0.1, halt_threshold=1.0
    )
    assert not result.newly_committed.any()
    assert not result.stop.any()
    logits[0, 0] = torch.tensor([12.0, -12.0])
    result = raw.alodlm_decision(
        logits,
        torch.full((1, 2), -10.0),
        active,
        absent,
        torch.zeros((1, 2)),
        entropy_threshold=0.1,
        halt_threshold=0.9,
    )
    assert result.newly_committed.tolist() == [[True, False]]
    assert result.unresolved.tolist() == [[False, True]]
    assert not result.stop.any()


def test_terminal_halt_uses_single_progress_fallback_not_all_token_commit():
    logits = torch.tensor([[[0.0, 0.0], [0.3, 0.0]]])
    active = torch.ones((1, 2), dtype=torch.bool)
    result = raw.alodlm_decision(
        logits,
        torch.full((1, 2), -100.0),
        active,
        torch.zeros_like(active),
        torch.zeros((1, 2)),
        entropy_threshold=0.0,
        halt_threshold=0.9,
        terminal=True,
    )
    assert result.stop.tolist() == [True]
    assert result.newly_committed.tolist() == [[False, True]]
    torch.testing.assert_close(result.cumulative_halt, torch.ones((1, 2)))
    assert result.unresolved.sum() == 1


def test_halt_accumulator_updates_active_rows_only_and_threshold_profiles_differ():
    active = torch.tensor([[True, False]])
    result = raw.alodlm_decision(
        torch.zeros((1, 2, 2)),
        torch.zeros((1, 2)),
        active,
        ~active,
        torch.tensor([[0.2, 0.7]]),
        entropy_threshold=0.0,
        halt_threshold=0.8,
    )
    torch.testing.assert_close(result.cumulative_halt, torch.tensor([[0.6, 0.7]]))
    args = (
        torch.zeros((1, 1, 1)),
        torch.zeros((1, 1)),
        torch.ones((1, 1), dtype=torch.bool),
        torch.zeros((1, 1), dtype=torch.bool),
        torch.zeros((1, 1)),
    )
    inclusive = raw.alodlm_decision(*args, entropy_threshold=0.0, halt_threshold=1.0)
    strict = raw.alodlm_decision(*args, entropy_threshold=0.0, halt_threshold=1.0, strict_entropy=True)
    assert inclusive.newly_committed.item()
    assert not strict.newly_committed.item()


def test_practical_objective_and_decision_execute_on_meta():
    with torch.device("meta"):
        model = raw.ALoDLMModel(config(), spec())
        ids = torch.empty((1, 2), dtype=torch.long)
        batch = raw.build_alodlm_batch(
            ids, ids, torch.empty_like(ids, dtype=torch.bool), 0.5, block_size=2, mask_token_id=0
        )
        output = model.rollout_training(batch, exit_depths=((0, 0, 1, 3),))
        objective = raw.alodlm_training_objective(output, batch)
        objective.loss.backward()
        result = raw.alodlm_decision(
            torch.empty((1, 2, 11)),
            torch.empty((1, 2)),
            torch.empty((1, 2), dtype=torch.bool),
            torch.empty((1, 2), dtype=torch.bool),
            torch.empty((1, 2)),
            entropy_threshold=0.5,
            halt_threshold=0.9,
        )
    assert objective.loss.device.type == "meta"
    assert result.stop.shape == (1,)
    assert model.exit_gate.projection.weight.grad is not None


@pytest.mark.parametrize("probability", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_selected_tensor_mask_probabilities_fail(probability):
    ids = torch.tensor([[1, 2]])
    with pytest.raises((RuntimeError, ValueError)):
        raw.build_alodlm_batch(
            ids,
            ids,
            torch.ones_like(ids, dtype=torch.bool),
            torch.full((1, 2), probability),
            block_size=2,
            mask_token_id=0,
        )


def test_masked_ignored_label_and_invalid_training_exit_fail():
    ids = torch.tensor([[1, 2]])
    with pytest.raises((RuntimeError, ValueError)):
        raw.build_alodlm_batch(
            ids, torch.tensor([[-100, 2]]), torch.ones_like(ids, dtype=torch.bool), 0.5, block_size=2, mask_token_id=0
        )
    model = raw.ALoDLMModel(config(), spec())
    for depths in (((0, 0, 0, 3),), ((0, 0, 4, 3),)):
        with pytest.raises((RuntimeError, ValueError)):
            model.rollout_training(make_batch(), exit_depths=depths)


@pytest.mark.parametrize(
    "changes",
    [
        {"feedback_policy": "unknown"},
        {"loop_start": 2, "loop_end": 2},
        {"max_depth": 0},
        {"gate_init_mass": (0.5, 0.5)},
    ],
)
def test_invalid_alodlm_profiles_fail(changes):
    with pytest.raises(ValueError):
        spec(**changes)


@pytest.mark.parametrize("loop_start,loop_end,depth", [(0, 3, 1), (0, 1, 2), (2, 3, 3)])
def test_empty_prelude_or_coda_and_single_depth_are_supported(loop_start, loop_end, depth):
    model = raw.ALoDLMModel(config(), spec(loop_start=loop_start, loop_end=loop_end, max_depth=depth, qk_norm=False))
    output = model.rollout_training(make_batch(), exit_depths=((0, 0, depth, depth),))
    assert output.logits.shape == (depth, 1, 4, 11)
    assert model.layer_evaluations() == loop_start + depth * (3 - loop_start)
    assert all(isinstance(layer.self_attn.q_norm, torch.nn.Identity) for layer in model.backbone.layers)
    result = raw.alodlm_training_objective(output, make_batch())
    if depth == 1:
        torch.testing.assert_close(result.exit_probabilities[0], make_batch().masked.float())
        assert result.actor.item() == 0
    result.loss.backward()
    assert all(parameter.grad is None or parameter.grad.isfinite().all() for parameter in model.parameters())


def test_published_partition_call_and_logical_buffer_counts_are_not_full_repeats():
    model = raw.ALoDLMModel(config(num_hidden_layers=36), spec(loop_start=10, loop_end=26, max_depth=4))
    assert model.layer_evaluations() == 114
    assert model.logical_buffer_count() == 84
    assert model.layer_evaluations(steps=2) == 62


@pytest.mark.parametrize("plan", [((1, 1, 3),), ((0, -1, 3),), ((0, 1, 4),)])
def test_invalid_inference_commit_plans_fail_explicitly(plan):
    model = raw.ALoDLMModel(config(), spec())
    with pytest.raises((ValueError, RuntimeError)):
        model.run_outer(
            torch.tensor([[4, 0, 0]]),
            torch.tensor([[0, 1, 2]]),
            torch.ones((3, 3), dtype=torch.bool).tril(),
            torch.tensor([[False, True, True]]),
            commit_depths=plan,
        )


def test_inference_cache_is_rejected_and_carry_normalization_flag_changes_recurrence():
    normalized = raw.ALoDLMModel(config(), spec(normalize_carry=True))
    unnormalized = raw.ALoDLMModel(config(), spec(normalize_carry=False))
    unnormalized.load_state_dict(normalized.state_dict(), strict=True)
    args = (
        torch.tensor([[4, 0, 0]]),
        torch.tensor([[0, 1, 2]]),
        torch.ones((3, 3), dtype=torch.bool).tril(),
        torch.tensor([[False, True, True]]),
    )
    kwargs = dict(commit_depths=((0, 0, 0),))
    first = normalized.run_outer(*args, **kwargs)
    second = unnormalized.run_outer(*args, **kwargs)
    torch.testing.assert_close(first.logits[0], second.logits[0], atol=0, rtol=0)
    assert not torch.allclose(first.logits[1], second.logits[1])
    cache = BlockDiffusionModel(config())(torch.tensor([[1]]), phase="prefill").cache
    with pytest.raises((ValueError, NotImplementedError)):
        normalized.run_outer(*args, **kwargs, cache=cache)


def test_vocab_exclusion_with_negative_infinity_has_zero_entropy_and_commits():
    result = raw.alodlm_decision(
        torch.tensor([[[0.0, -torch.inf]]]),
        torch.tensor([[-10.0]]),
        torch.ones((1, 1), dtype=torch.bool),
        torch.zeros((1, 1), dtype=torch.bool),
        torch.zeros((1, 1)),
        entropy_threshold=0.0,
        halt_threshold=0.9,
    )
    assert result.entropy.item() == 0.0
    assert result.entropy.isfinite().all()
    assert result.newly_committed.item()
    assert result.tokens.item() == 0


@pytest.mark.parametrize("scores", [[float("nan"), 0.0], [float("inf"), 0.0], [-float("inf"), -float("inf")]])
def test_decision_rejects_invalid_vocabulary_distributions(scores):
    with pytest.raises((RuntimeError, ValueError)):
        raw.alodlm_decision(
            torch.tensor([[scores]]),
            torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.bool),
            torch.zeros((1, 1)),
            entropy_threshold=0.5,
            halt_threshold=0.9,
        )


def test_vocab_masked_decision_meta_capture_does_not_read_values():
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch.device("meta"):
        scores = torch.empty((1, 2, 3))
        excluded = torch.empty((1, 2, 3), dtype=torch.bool)
        gate = torch.empty((1, 2))
        unresolved = torch.empty((1, 2), dtype=torch.bool)
        committed = torch.empty((1, 2), dtype=torch.bool)
        halt = torch.empty((1, 2))
    graph = make_fx(
        lambda values: raw.alodlm_decision(
            values.masked_fill(excluded, -torch.inf),
            gate,
            unresolved,
            committed,
            halt,
            entropy_threshold=0.5,
            halt_threshold=0.9,
        ).entropy
    )(scores)
    assert graph(scores).shape == (1, 2)
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)


def _custom_ignore_fixture():
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[1, 0, 3, 0]])
    masked = torch.tensor([[False, False, True, False]])
    custom = raw.build_alodlm_batch(ids, labels, masked, 0.5, block_size=2, mask_token_id=0, ignore_index=0)
    standard = raw.build_alodlm_batch(
        ids, labels.masked_fill(labels == 0, -100), masked, 0.5, block_size=2, mask_token_id=0
    )
    output = raw.ALoDLMOutput(
        torch.randn((3, 1, 8, 7), requires_grad=True),
        torch.randn((3, 1, 8), requires_grad=True),
        torch.full((1, 8), 3, dtype=torch.long),
        torch.zeros((1, 8, 8)),
    )
    return output, custom, standard


def test_objective_inherits_custom_ignore_index_from_batch():
    output, custom, standard = _custom_ignore_fixture()
    assert custom.ignore_index == 0
    assert standard.ignore_index == -100
    actual = raw.alodlm_training_objective(output, custom)
    expected = raw.alodlm_training_objective(output, standard)
    torch.testing.assert_close(actual.autoregressive, expected.autoregressive, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    actual_gradients = torch.autograd.grad(actual.loss, (output.logits, output.gate_logits), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.loss, (output.logits, output.gate_logits))
    for result, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


@pytest.mark.parametrize("override", [-100, True])
def test_objective_rejects_conflicting_or_boolean_ignore_index(override):
    output, batch, _standard = _custom_ignore_fixture()
    with pytest.raises(ValueError):
        raw.alodlm_training_objective(output, batch, ignore_index=override)


def test_all_ignored_ar_with_custom_zero_index_has_zero_loss_and_gradients():
    ids = torch.tensor([[1, 2]])
    batch = raw.build_alodlm_batch(
        ids,
        torch.zeros_like(ids),
        torch.zeros_like(ids, dtype=torch.bool),
        0.5,
        block_size=2,
        mask_token_id=0,
        ignore_index=0,
    )
    output = raw.ALoDLMOutput(
        torch.randn((3, 1, 4, 7), requires_grad=True),
        torch.randn((3, 1, 4), requires_grad=True),
        torch.full((1, 4), 3, dtype=torch.long),
        torch.zeros((1, 4, 8)),
    )
    objective = raw.alodlm_training_objective(output, batch)
    assert objective.autoregressive.item() == objective.loss.item() == 0.0
    objective.loss.backward()
    assert torch.count_nonzero(output.logits.grad) == 0
    assert torch.count_nonzero(output.gate_logits.grad) == 0
