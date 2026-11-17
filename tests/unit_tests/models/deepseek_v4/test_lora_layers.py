# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from functools import partial
import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.common import Linear, decoder_sharding, moe
from torchtitan.protocols import sharding
from torchtitan_npu.models import deepseek_v4 as dsv4
from torchtitan_npu.models.deepseek_v4 import lora
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.fixture(params=[False, True], ids=["eager", "selective-ac"])
def forward_with_checkpoint(request, monkeypatch):
    if not request.param:
        return lambda module, *args, **kwargs: module(*args, **kwargs)
    from torch.utils.checkpoint import DefaultDeviceType, checkpoint, create_selective_checkpoint_contexts

    monkeypatch.setattr(DefaultDeviceType, "get_device_type", lambda: "cpu")
    context_fn = partial(create_selective_checkpoint_contexts, list(SelectiveAC.Config().build().get_save_ops()))
    return partial(checkpoint, use_reentrant=False, context_fn=context_fn)


def _assert_gradients_match(actual, expected, variables, *, rtol=1e-5, atol=1e-6):
    upstream = torch.linspace(-0.7, 0.9, actual.numel()).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, variables, upstream)
    expected_grads = torch.autograd.grad(expected, variables, upstream)
    for result, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)


def _grouped_mm(*, A, B_t, offs):
    outputs = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        outputs.append(A[start:end] @ B_t[expert])
        start = end
    return torch.cat(outputs) if outputs else A.new_empty((0, B_t.shape[-1]))


_GroupedLoRAExperts = lora.grouped_lora_class(moe.GroupedExperts)


class _CpuGroupedLoRAExperts(_GroupedLoRAExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(_GroupedLoRAExperts.Config):
        pass

    def _grouped_mm(self, *, A, B_t, offs):
        return _grouped_mm(A=A, B_t=B_t, offs=offs)


def _build_cpu_grouped_lora_module() -> _CpuGroupedLoRAExperts:
    config = _CpuGroupedLoRAExperts.Config(
        dim=4, hidden_dim=6, num_experts=2, rank=2, alpha=4.0, chunk_rows=2,
        param_init=dict.fromkeys(
            ("w1_EFD", "w2_EDF", "w3_EFD", "w13_lora_a", "w1_lora_b", "w3_lora_b", "w2_lora_a", "w2_lora_b"), torch.nn.init.zeros_
        ),
    )
    module = config.build()
    for parameter in (module.w1_EFD, module.w2_EDF, module.w3_EFD):
        parameter.requires_grad_(False)
    module.init_states()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    return module


def _per_expert_lora_reference(module, inputs, counts, scores, limit):
    expected_parts = []
    start = 0
    scale = 2.0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        x = inputs[start:end].bfloat16()
        w13_hidden = x @ module.w13_lora_a[expert].bfloat16().T
        gate_delta = w13_hidden @ module.w1_lora_b[expert].bfloat16().T
        up_delta = w13_hidden @ module.w3_lora_b[expert].bfloat16().T
        gate = x @ module.w1_EFD[expert].bfloat16().T
        gate = gate + scale * gate_delta
        up = x @ module.w3_EFD[expert].bfloat16().T
        up = up + scale * up_delta
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        hidden = F.silu(gate) * up
        hidden = (hidden.float() * scores[start:end, None]).to(hidden.dtype)
        output = hidden @ module.w2_EDF[expert].bfloat16().T
        output = output + scale * (
            (hidden @ module.w2_lora_a[expert].bfloat16().T) @ module.w2_lora_b[expert].bfloat16().T
        )
        expected_parts.append(output.float())
        start = end
    return torch.cat(expected_parts)


def test_indexer_scores_and_gradients_with_checkpoint(forward_with_checkpoint):
    query = torch.randn(1, 4, 2, 3, requires_grad=True)
    key = torch.randn(1, 2, 3, requires_grad=True)
    weight = torch.randn(1, 4, 2, requires_grad=True)
    picks = torch.tensor([[0, 1]] * 4, dtype=torch.int64).view(1, 4, 2)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 2], dtype=torch.int32)
    actual = forward_with_checkpoint(
        dsv4.compressor.student_logits_at_picks, query, key, weight, picks, cu_seqlens, cu_seqlens_k
    )
    expected = torch.stack(
        [
            (torch.einsum("hd,d->h", query[0, token], key[0, pick]).relu() * weight[0, token]).sum()
            for token in range(4)
            for pick in (0, 1)
        ]
    ).view(1, 4, 2)
    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(actual, expected, (query, key, weight))


def test_replicated_base_keeps_both_lora_factors_replicated():
    replicated = decoder_sharding.dense_param_placement(tp=spmd.R)
    base = sharding.ShardingConfig(state_shardings={"weight": replicated})

    lora_a, lora_b = lora.adapter_sharding(base)

    assert lora_a is not None and lora_a.state_shardings["weight"] == replicated
    assert lora_b is not None and lora_b.state_shardings["weight"] == replicated


def test_chunked_dense_lora_matches_independent_formula_and_backpropagates(forward_with_checkpoint):
    cls = lora.linear_lora_class(Linear)
    assert cls is lora.linear_lora_class(Linear)
    module = cls.Config(in_features=5, out_features=7, bias=False, rank=3, alpha=6.0, chunk_rows=2).build()
    module.weight.requires_grad_(False)
    module.init_states()
    with torch.no_grad():
        module.lora_b.weight.normal_(std=0.1)

    inputs = torch.randn(2, 3, 5, requires_grad=True)
    actual = forward_with_checkpoint(module, inputs)
    expected = F.linear(inputs, module.weight) + 2.0 * F.linear(
        F.linear(inputs, module.lora_a.weight), module.lora_b.weight
    )

    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(actual, expected, (inputs, module.lora_a.weight, module.lora_b.weight))
    assert not module.weight.requires_grad


def test_batched_lora_matches_shared_a_per_head_b_and_backpropagates(forward_with_checkpoint):
    cls = lora.linear_lora_class(BatchedLinear)
    assert cls is lora.linear_lora_class(BatchedLinear)
    module = cls.Config(
        n_heads=3, in_features=5, out_features=4, rank=2, alpha=4.0, chunk_rows=2,
        param_init={"weight": torch.nn.init.zeros_},
    ).build()
    module.weight.requires_grad_(False)
    module.init_states()
    with torch.no_grad():
        module.weight.normal_(std=0.1)
        module.lora_b.weight.normal_(std=0.1)

    inputs = torch.randn(2, 3, 3, 5, requires_grad=True)
    actual = forward_with_checkpoint(module, inputs)
    base_weight = module.weight.view(3, 4, 5)
    adapter_b = module.lora_b.weight.view(3, 4, 2)
    expected = torch.einsum("...hd,hod->...ho", inputs, base_weight)
    hidden = F.linear(inputs, module.lora_a.weight)
    expected = expected + 2.0 * torch.einsum("...hr,hor->...ho", hidden, adapter_b)

    torch.testing.assert_close(actual, expected)
    _assert_gradients_match(actual, expected, (inputs, module.lora_a.weight, module.lora_b.weight))
    assert not module.weight.requires_grad


@pytest.mark.parametrize("limit", [0.0, 0.05], ids=["unclamped", "clamped"])
def test_grouped_lora_output_and_gradients_match_per_expert_reference(limit):
    assert _GroupedLoRAExperts is lora.grouped_lora_class(moe.GroupedExperts)
    module = _build_cpu_grouped_lora_module()
    module.swiglu_limit = limit
    counts = torch.tensor([2, 3], dtype=torch.int64)
    inputs = torch.randn(5, 4, requires_grad=True)
    scores = torch.linspace(0.2, 1.4, 5, requires_grad=True)

    actual = module(inputs, counts, routed_scores_R=scores)
    expected = _per_expert_lora_reference(module, inputs, counts, scores, limit)
    variables = (inputs, scores, module.w13_lora_a, module.w1_lora_b, module.w3_lora_b, module.w2_lora_a, module.w2_lora_b)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    _assert_gradients_match(actual, expected, variables, rtol=2e-2, atol=2e-3)
    assert all(not parameter.requires_grad for name, parameter in module.named_parameters() if "lora_" not in name)


@pytest.mark.parametrize("freeze_lora_a", [False, True], ids=["ab-training", "b-only-training"])
def test_dense_lora_zero_init_preserves_output_and_updates_only_adapter(freeze_lora_a):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        target_modules=["attention.wo_b"], adapt_routed_experts=False, strict=False
    )
    model_config = dsv4.model_registry("debugmodel", converters=[converter]).model
    configs = model_config.traverse(Linear.Config, recurse=True)
    target_config = next(config for fqn, config, _, _ in configs if fqn == "layers.0.attention.wo_b")
    target = target_config.build()
    target.weight.requires_grad_(False)
    target.init_states()
    inputs = torch.linspace(-1.0, 1.0, steps=2 * target.in_features).reshape(2, target.in_features)

    expected = torch.nn.functional.linear(inputs, target.weight)
    actual = target(inputs)

    assert torch.count_nonzero(target.lora_a.weight) > 0
    assert torch.count_nonzero(target.lora_b.weight) == 0
    torch.testing.assert_close(actual, expected)

    target.lora_a.weight.requires_grad_(not freeze_lora_a)
    adapter_a_before = target.lora_a.weight.detach().clone()
    trainable = [parameter for parameter in target.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.1, foreach=False)
    base_before = target.weight.detach().clone()
    adapter_before = target.lora_b.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        target(inputs).square().mean().backward()

        assert all(parameter.grad is not None for parameter in trainable)
        assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in trainable)
        if freeze_lora_a:
            assert target.lora_a.weight.grad is None

        optimizer.step()

        assert torch.equal(target.weight, base_before)
        assert not torch.equal(target.lora_b.weight, adapter_before)
        if freeze_lora_a:
            assert torch.equal(target.lora_a.weight, adapter_a_before)
    if not freeze_lora_a:
        assert not torch.equal(target.lora_a.weight, adapter_a_before)


def test_grouped_legacy_weights_load_into_split_gate_up():
    module = _build_cpu_grouped_lora_module()
    original = {name: value.clone() for name, value in module.state_dict().items()}
    legacy = dict(original)
    legacy["w13_lora_b"] = torch.stack((legacy.pop("w1_lora_b"), legacy.pop("w3_lora_b")), dim=2)
    restored = _build_cpu_grouped_lora_module()

    restored.load_state_dict(legacy, strict=True)

    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
