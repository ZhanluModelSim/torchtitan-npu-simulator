# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check real training execution and captured operator/storage accounting."""

import json
import math

import pytest
import torch
import training_workload
from model import BlockDiffusionModel, ModelConfig, TrainingSpec
from training_workload import capture_training_step


def config(**updates):
    values = dict(
        vocab_size=13,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        num_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=6,
        num_shared_experts=0,
        dtype=torch.float32,
    )
    values.update(updates)
    return ModelConfig(**values)


def capture(**updates):
    kwargs = dict(
        config=config(),
        spec=TrainingSpec(block_size=2),
        batch_size=1,
        seq_len=4,
        num_masked_tokens_per_request=2,
        device="cpu",
        optimizer="none",
    )
    kwargs.update(updates)
    return capture_training_step(**kwargs)


def storage_bytes(tensors):
    return sum({item.storage_id: item.storage_nbytes for item in tensors}.values())


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_trace_contains_real_forward_backward_and_unknown_costs(device):
    report = capture(device=device)
    assert report.attention_backend == "pytorch_sdpa_math"
    phases = {record.phase for record in report.records}
    assert phases == {"forward", "backward"}
    backward = [record for record in report.records if record.phase == "backward"]
    assert any("backward" in record.op for record in backward)
    assert any(record.op in ("aten.mm.default", "aten.bmm.default") for record in backward)
    assert any(record.known_matrix_flops is None for record in backward)
    assert report.optimizer_state_bytes == 0
    assert report.optimizer_config == {}
    assert report.valid_tokens == 2
    if device == "cpu":
        assert math.isfinite(report.loss)
    else:
        assert report.loss is None
        assert not any("_local_scalar_dense" in record.op for record in report.records)
    assert report.parameter_bytes == storage_bytes(report.parameter_tensors)
    assert report.gradient_bytes == storage_bytes(report.gradient_tensors)
    assert report.gradient_bytes > 0
    data = json.loads(json.dumps(report.to_dict()))
    assert data["scope"] == "actual_single_rank_training_reference_trace"
    assert data["phases"]["backward"]["num_operators"] == len(backward)
    assert data["phases"]["backward"]["num_operators_without_matrix_flop_model"] > 0


def test_matrix_flop_counts_come_from_actual_recorded_shapes():
    report = capture()
    checked = {"forward": 0, "backward": 0}
    for record in report.records:
        if record.known_matrix_flops is None:
            continue
        assert record.op in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default")
        if record.op == "aten.addmm.default":
            left, right = record.inputs[1:3]
        else:
            left, right = record.inputs[:2]
        assert left.shape[-1] == right.shape[-2]
        assert record.outputs[0].shape[-2:] == (left.shape[-2], right.shape[-1])
        expected = 2 * math.prod(record.outputs[0].shape) * left.shape[-1]
        assert record.known_matrix_flops == expected
        checked[record.phase] += 1
    assert checked["forward"] >= 7
    assert checked["backward"] >= 7


def test_tensor_identity_and_aliasing_are_recorded_not_inferred_from_shapes():
    report = capture()
    assert [record.sequence for record in report.records] == list(range(len(report.records)))
    all_ids = set()
    alias_found = False
    for record in report.records:
        for item in record.inputs + record.outputs:
            assert len(item.shape) == len(item.stride)
            assert item.storage_offset >= 0
            all_ids.add(item.tensor_id)
        for source in record.inputs:
            for result in record.outputs:
                if source.tensor_id != result.tensor_id and source.storage_id == result.storage_id:
                    alias_found = True
        assert set(record.mutated_inputs).issubset({item.tensor_id for item in record.inputs})
    assert alias_found
    assert len(all_ids) > len(report.parameter_tensors)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_adamw_operators_and_state_are_materialized_by_real_update(device):
    report = capture(device=device, optimizer="adamw")
    optimizer_ops = [record for record in report.records if record.phase == "optimizer"]
    assert optimizer_ops
    settings = report.to_dict()["optimizer_config"]
    assert settings["lr"] == 1e-3
    assert settings["betas"] == [0.9, 0.999]
    assert settings["eps"] == 1e-8
    assert settings["weight_decay"] == 0.01
    assert settings["foreach"] is False
    assert settings["fused"] is False
    assert any(record.mutated_inputs for record in optimizer_ops)
    assert report.optimizer_state_bytes == storage_bytes(report.optimizer_state_tensors)
    assert report.optimizer_state_bytes > 2 * report.gradient_bytes  # Two moments plus scalar step tensors.
    assert any(item.shape == () for item in report.optimizer_state_tensors)
    assert any(item.shape != () for item in report.optimizer_state_tensors)


@pytest.mark.parametrize("tied", [False, True])
def test_parameter_storage_matches_core_and_deduplicates_shared_weights(tied):
    cfg = config(tie_word_embeddings=tied)
    report = capture(config=cfg)
    reference = BlockDiffusionModel(cfg)
    assert report.parameter_bytes == reference.parameter_memory_bytes()
    assert len({item.storage_id for item in report.parameter_tensors}) == len(report.parameter_tensors)


def test_inactive_experts_do_not_receive_fabricated_gradients_or_optimizer_state(monkeypatch):
    original = training_workload.BlockDiffusionModel
    models = []

    def retain(cfg):
        model = original(cfg)
        models.append(model)
        return model

    monkeypatch.setattr(training_workload, "BlockDiffusionModel", retain)
    report = capture(config=config(num_experts=8), seq_len=2, num_masked_tokens_per_request=2, optimizer="adamw")
    model = models[0]
    assert all(
        parameter.grad is None for expert in model.layers[0].mlp.experts[2:] for parameter in expert.parameters()
    )
    expected_gradients = tuple(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    assert report.gradient_bytes == sum(tensor.numel() * tensor.element_size() for tensor in expected_gradients)
    assert report.gradient_bytes < report.parameter_bytes
    assert len(report.gradient_tensors) == len(expected_gradients)
    assert report.optimizer_state_bytes == 2 * report.gradient_bytes + 4 * len(expected_gradients)


def test_frozen_parameter_accounting_uses_actual_gradients(monkeypatch):
    original = training_workload.BlockDiffusionModel
    models = []

    def freeze_head(cfg):
        model = original(cfg)
        model.lm_head.weight.requires_grad_(False)
        models.append(model)
        return model

    monkeypatch.setattr(training_workload, "BlockDiffusionModel", freeze_head)
    report = capture(optimizer="adamw")
    model = models[0]
    assert model.lm_head.weight.grad is None
    actual = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert report.gradient_bytes == sum(grad.numel() * grad.element_size() for grad in actual)
    assert report.parameter_bytes - report.gradient_bytes == model.lm_head.weight.numel() * 4
    assert report.optimizer_state_bytes == 2 * report.gradient_bytes + 4 * len(actual)


def test_empty_training_targets_still_capture_actual_backward():
    report = capture(num_masked_tokens_per_request=0)
    assert report.loss == 0
    assert report.valid_tokens == 0
    assert any(record.phase == "backward" for record in report.records)
    assert report.gradient_bytes > 0  # Zero-valued dense gradients are still materialized.


def test_reference_capture_does_not_change_callers_random_state():
    torch.manual_seed(819)
    before = torch.random.get_rng_state().clone()
    capture()
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "updates",
    [
        {"device": "cuda"},
        {"optimizer": "sgd"},
        {"batch_size": 0},
        {"num_masked_tokens_per_request": -1},
        {"num_masked_tokens_per_request": 5},
        {"seq_len": 3},
        {"config": config(vocab_size=2), "spec": TrainingSpec(block_size=2, ignore_index=1)},
        {
            "spec": TrainingSpec(block_size=2, attention_policy="prefix_causal_canvas_bidirectional"),
            "num_masked_tokens_per_request": 3,
        },
    ],
)
def test_capture_rejects_unsupported_or_invalid_workloads(updates):
    with pytest.raises(ValueError):
        capture(**updates)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_synthetic_targets_exclude_in_vocab_ignore_index_consistently(device):
    report = capture(
        device=device,
        spec=TrainingSpec(block_size=2, ignore_index=1),
        batch_size=2,
        num_masked_tokens_per_request=2,
    )
    assert report.valid_tokens == 4
    if device == "cpu":
        assert math.isfinite(report.loss)
