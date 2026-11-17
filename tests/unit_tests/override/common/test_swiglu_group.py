# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def swiglu_module():
    # Apply the plugin's class replacements before importing their consumers.
    importlib.import_module("torchtitan_npu")
    return importlib.import_module("torchtitan_npu.override.common.swiglu_group")


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture
def cpu_swiglu_boundary(monkeypatch, swiglu_module):
    """Differentiable CPU substitute for CANN, not a kernel numerics test."""
    calls = []

    def swiglu(h, *, weight, group_index, clamp_limit):
        calls.append((h, weight, group_index, clamp_limit))
        gate, up = h.chunk(2, dim=-1)
        if clamp_limit > 0:
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(min=-clamp_limit, max=clamp_limit)
        hidden = F.silu(gate) * up
        if weight is not None:
            hidden = (hidden.float() * weight.reshape(-1, 1)).to(h.dtype)
        return hidden

    monkeypatch.setattr(torch.ops, "cann_ops_nn", SimpleNamespace(swiglu_group=SimpleNamespace(default=swiglu)))
    monkeypatch.setattr(swiglu_module.ascendc, "get_spmd_backend", lambda: "torch")
    return calls


@pytest.mark.parametrize("target", ["routed", "shared"])
def test_swiglu_override_imports_replace_only_requested_experts(monkeypatch, swiglu_module, target):
    # Resolve the registry only after the plugin patches have been applied.
    from torchtitan.config import OverrideConfig, apply_overrides

    from torchtitan_npu.models.deepseek_v4 import model_registry

    monkeypatch.setitem(sys.modules, "cann_ops_nn.ops", ModuleType("cann_ops_nn.ops"))
    model = model_registry("debugmodel").model
    originals = [
        (layer.moe.routed_experts.inner_experts, layer.moe.shared_experts, layer.attention) for layer in model.layers
    ]
    factory = "asc" if target == "routed" else "asc_shared_experts"

    replacements = apply_overrides(
        OverrideConfig(imports=[f"torchtitan_npu.override.common.swiglu_group.{factory}"]), model
    )

    assert len(replacements) == len(model.layers)
    for layer, (routed, shared, attention) in zip(model.layers, originals, strict=True):
        assert layer.attention is attention
        if target == "routed":
            selected = layer.moe.routed_experts.inner_experts
            assert isinstance(selected, swiglu_module.AscGroupedExperts.Config)
            assert layer.moe.shared_experts is shared
        else:
            selected = layer.moe.shared_experts
            assert isinstance(selected, swiglu_module.AscFeedForward.Config)
            assert layer.moe.routed_experts.inner_experts is routed


@pytest.fixture
def cpu_grouped_swiglu_boundary(monkeypatch, swiglu_module, cpu_swiglu_boundary):
    def grouped_mm(x, weight, *, offs):
        outputs = []
        start = 0
        for expert, end in enumerate(offs.tolist()):
            outputs.append(x[start:end] @ weight[expert])
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(swiglu_module.ascendc, "_grouped_mm", grouped_mm)
    return cpu_swiglu_boundary


def dense_grouped_experts(x_ref, scores_ref, weights, counts, limit):
    """Independent per-expert reference using dense linear operations."""
    w1, w2, w3 = weights
    expected_rows = []
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        rows = x_ref[start:end]
        # Separate casts accumulate gate/up input gradients in FP32, as in forward.
        gate = F.linear(rows.bfloat16(), w1[expert].bfloat16())
        up = F.linear(rows.bfloat16(), w3[expert].bfloat16())
        if limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        hidden = F.silu(gate) * up
        if scores_ref is not None:
            hidden = (hidden.float() * scores_ref[start:end, None].float()).bfloat16()
        expected_rows.append(F.linear(hidden, w2[expert].bfloat16()))
        start = end
    return torch.cat(expected_rows).float()


@pytest.mark.parametrize(
    ("counts", "with_scores", "limit"),
    [((0, 0), True, 0.0), ((1, 0), False, -1.0), ((2, 1), True, 0.0), ((2, 1), True, 0.75), ((2, 1), False, 0.75)],
    ids=["empty_scored", "single_unscored_negative_limit", "uneven_scored", "clamped_scored", "clamped_unscored"],
)
def test_grouped_experts_output_and_gradients_match_dense(
    swiglu_module, cpu_grouped_swiglu_boundary, counts, with_scores, limit
):
    experts = swiglu_module.AscGroupedExperts.Config(dim=2, hidden_dim=4, num_experts=2, swiglu_limit=limit).build()
    # Mixed signs cross both up-clamp boundaries and the gate upper bound.
    w1_data = torch.tensor(
        [[[2.0, 0.5], [-2.0, 1.0], [0.5, -1.0], [1.0, 1.0]], [[1.0, -1.0], [2.0, 0.5], [-1.0, 2.0], [0.5, 1.0]]]
    )
    w3_data = torch.tensor(
        [[[1.0, -2.0], [-1.0, 0.5], [2.0, 1.0], [-0.5, -1.0]], [[-1.0, -1.0], [0.5, 2.0], [1.0, -0.5], [2.0, 1.0]]]
    )
    w2_data = torch.tensor(
        [[[0.5, -1.0, 0.25, 1.0], [-0.5, 0.25, 1.0, 0.5]], [[1.0, 0.5, -0.25, 0.5], [0.25, -1.0, 0.5, 1.0]]]
    )
    with torch.no_grad():
        for name, value in (("w1_EFD", w1_data), ("w2_EDF", w2_data), ("w3_EFD", w3_data)):
            getattr(experts, name).copy_(value)
    x_data = torch.tensor([[1.0, -0.5], [-0.5, 1.0], [0.75, 0.5]])[: sum(counts)]
    score_data = torch.tensor([[0.25, 9.0], [0.5, 9.0], [0.75, 9.0]], dtype=torch.float64)[: sum(counts), 0]
    x_ref = x_data.clone().requires_grad_()
    scores_ref = score_data.clone().requires_grad_() if with_scores else None
    w1, w2, w3 = [w.clone().requires_grad_() for w in (w1_data, w2_data, w3_data)]
    expected = dense_grouped_experts(x_ref, scores_ref, (w1, w2, w3), counts, limit)
    upstream = torch.tensor([[0.5, -1.0], [1.0, 0.25], [-0.5, 0.75]])[: sum(counts)]
    reference_inputs = (x_ref, w1, w2, w3) if scores_ref is None else (x_ref, w1, w2, w3, scores_ref)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream)

    x = x_data.clone().requires_grad_()
    scores = score_data.detach().requires_grad_() if with_scores else None
    actual = experts(x, torch.tensor(counts), routed_scores_R=scores)
    weights = (experts.w1_EFD, experts.w2_EDF, experts.w3_EFD)
    actual_inputs = (x, *weights) if scores is None else (x, *weights, scores)
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert len(cpu_grouped_swiglu_boundary) == 1
    packed, op_scores, group_index, clamp_limit = cpu_grouped_swiglu_boundary[0]
    assert packed.shape == (sum(counts), 8)
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    if with_scores:
        torch.testing.assert_close(op_scores, score_data.float())
        assert op_scores.is_contiguous()
    else:
        assert op_scores is None
    assert group_index is None
    assert clamp_limit == (limit if limit > 0 else -1.0)


@pytest.mark.parametrize("limit", [0.0, 0.75], ids=["unclamped", "clamped"])
def test_shared_experts_output_and_gradients_match_dense(swiglu_module, cpu_swiglu_boundary, limit):
    from torchtitan.models.common.linear import Linear

    configs = dict(
        w1=Linear.Config(in_features=2, out_features=4, bias=False),
        w2=Linear.Config(in_features=4, out_features=2, bias=False),
        w3=Linear.Config(in_features=2, out_features=4, bias=False),
        swiglu_limit=limit,
    )
    experts = swiglu_module.AscFeedForward.Config(**configs).build()
    w1_data = torch.tensor([[2.0, 0.5], [-2.0, 1.0], [0.5, -1.0], [1.0, 1.0]])
    w3_data = torch.tensor([[1.0, -2.0], [-1.0, 0.5], [2.0, 1.0], [-0.5, -1.0]])
    w2_data = torch.tensor([[0.5, -1.0, 0.25, 1.0], [-0.5, 0.25, 1.0, 0.5]])
    with torch.no_grad():
        for name, value in (("w1", w1_data), ("w2", w2_data), ("w3", w3_data)):
            getattr(experts, name).weight.copy_(value)
    x_data = torch.tensor([[[1.0, -0.5], [-0.5, 1.0]], [[0.75, 0.5], [-1.0, -0.25]]])
    x_ref = x_data.clone().requires_grad_()
    w1, w2, w3 = [w.clone().requires_grad_() for w in (w1_data, w2_data, w3_data)]
    gate, up = F.linear(x_ref, w1), F.linear(x_ref, w3)
    if limit > 0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    expected = F.linear(F.silu(gate) * up, w2)
    upstream = torch.tensor([[[0.5, -1.0], [1.0, 0.25]], [[-0.5, 0.75], [0.25, -0.5]]])
    expected_grads = torch.autograd.grad(expected, (x_ref, w1, w2, w3), upstream)

    x = x_data.clone().requires_grad_()
    actual = experts(x)
    actual_grads = torch.autograd.grad(actual, (x, experts.w1.weight, experts.w2.weight, experts.w3.weight), upstream)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert len(cpu_swiglu_boundary) == 1
    packed, op_scores, group_index, clamp_limit = cpu_swiglu_boundary[0]
    assert packed.shape == (2, 2, 8)
    assert packed.dtype == x_data.dtype
    assert op_scores is None
    assert group_index is None
    assert clamp_limit == (limit if limit > 0 else -1.0)
