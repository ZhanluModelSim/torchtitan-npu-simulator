# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, MoE

from torchtitan_npu.converters.kernels import gmm as gmm_module
from torchtitan_npu.converters.kernels import moe_dispatch as moe_module
from torchtitan_npu.converters.kernels import swiglu_group as swiglu_module
from torchtitan_npu.converters.framework.model_custom_config_converter import (
    ModelCustomConfigConverter,
)
from torchtitan_npu.models.common import moe as common_moe

_EXPERT_ACTIVATION_ATTR = "_expert_activation"
_EXPERT_ACTIVATION_FN_ATTR = "_expert_activation_fn"


def _install_fake_ops(monkeypatch, forward, backward=None):
    namespace = SimpleNamespace(
        swiglu_group=SimpleNamespace(default=forward),
        swiglu_group_backward=SimpleNamespace(default=backward),
    )
    monkeypatch.setattr(torch.ops, "cann_ops_nn", namespace)


def _slice_swiglu_output(x, *, weight, group_index, clamp_limit):
    return x[..., : x.shape[-1] // 2].clone()


def _allow_cann_ops_import(monkeypatch):
    monkeypatch.setattr(swiglu_module.importlib, "import_module", lambda _name: None)


def _grouped_experts() -> GroupedExperts:
    return GroupedExperts.Config(
        dim=2,
        hidden_dim=2,
        num_experts=1,
        use_grouped_mm=False,
    ).build()


def _feed_forward(dim: int = 2, hidden_dim: int = 2) -> FeedForward:
    return FeedForward.Config(
        w1=Linear.Config(in_features=dim, out_features=hidden_dim, bias=False),
        w2=Linear.Config(in_features=hidden_dim, out_features=dim, bias=False),
        w3=Linear.Config(in_features=dim, out_features=hidden_dim, bias=False),
    ).build()


def _swiglu_group_converter():
    return swiglu_module.NpuSwigluGroupConverter(SimpleNamespace(name="deepseek_v4"))


def _moe_dispatch_converter():
    return moe_module.NpuMoeDispatchConverter(SimpleNamespace(name="deepseek_v4"))


class _IdentityStateDictAdapter:
    @classmethod
    def to_hf(cls, state_dict):
        return state_dict

    @classmethod
    def from_hf(cls, state_dict):
        return state_dict


def _apply_model_config(model, model_spec, model_config):
    class TestModelConfigConverter(ModelCustomConfigConverter):
        _model_config = model_config

    converter = TestModelConfigConverter.__new__(TestModelConfigConverter)
    converter.model_spec = model_spec
    converter.convert(model)


def _moe_with_shared_experts() -> MoE:
    module = MoE.__new__(MoE)
    nn.Module.__init__(module)
    module.shared_experts = _feed_forward()
    return module


class _SharedExpertContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = _feed_forward()
        self.moe = _moe_with_shared_experts()


class _FeedForwardContainer(_SharedExpertContainer):
    def __init__(self):
        super().__init__()
        self.experts = _grouped_experts()


def test_activation_calls_documented_op_with_gmm_inputs(monkeypatch):
    calls = []

    def forward(x, *, weight, group_index, clamp_limit):
        calls.append((x, weight, group_index, clamp_limit))
        return x[..., : x.shape[-1] // 2]

    _install_fake_ops(monkeypatch, forward)
    h = torch.randn(4, 16)
    routed_scores = torch.randn(4, 2, dtype=torch.float64)[:, :1]

    output = swiglu_module.swiglu_group_activation(h, 3.0, routed_scores)

    assert output.shape == (4, 8)
    op_x, op_weight, group_index, clamp_limit = calls[0]
    assert op_x is h
    assert op_weight.dtype == torch.float32
    assert op_weight.is_contiguous()
    assert group_index is None
    assert clamp_limit == 3.0


def test_activation_uses_documented_optional_defaults(monkeypatch):
    calls = []

    def forward(x, *, weight, group_index, clamp_limit):
        calls.append((weight, group_index, clamp_limit))
        return x[..., : x.shape[-1] // 2]

    _install_fake_ops(monkeypatch, forward)

    swiglu_module.swiglu_group_activation(torch.randn(4, 16))

    assert calls == [(None, None, -1.0)]


def test_activation_delegates_autograd_to_documented_op(monkeypatch):
    def forward(x, *, weight, group_index, clamp_limit):
        assert group_index is None
        assert clamp_limit == -1.0
        return x[..., : x.shape[-1] // 2] * weight

    def fail_backward(*args, **kwargs):
        raise AssertionError("local Autograd bridge must not call the backward op")

    _install_fake_ops(monkeypatch, forward, fail_backward)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    routed_scores = torch.tensor([[2.0], [3.0]], requires_grad=True)

    swiglu_module.swiglu_group_activation(x, routed_scores=routed_scores).sum().backward()

    assert torch.equal(
        x.grad,
        torch.tensor([[2.0, 2.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]]),
    )
    assert torch.equal(routed_scores.grad, torch.tensor([[1.0], [9.0]]))


def test_converter_imports_cann_ops_without_executing_a_runtime_probe(monkeypatch):
    model = _SharedExpertContainer()
    imports = []

    monkeypatch.setattr(swiglu_module.importlib, "import_module", imports.append)
    assert not hasattr(swiglu_module, "_probe_swiglu_group_execution")

    _swiglu_group_converter().convert(model)

    assert imports == ["cann_ops_nn.ops"]


def test_converter_propagates_cann_ops_import_error(monkeypatch):
    model = _FeedForwardContainer()
    routed_experts = model.experts
    routed_weights = (routed_experts.w1, routed_experts.w2, routed_experts.w3)
    shared_experts = model.moe.shared_experts
    shared_linears = (shared_experts.w1, shared_experts.w2, shared_experts.w3)
    state_dict_keys = tuple(model.state_dict())

    def raise_import_error(_name):
        raise ModuleNotFoundError("No module named 'cann_ops_nn'")

    monkeypatch.setattr(swiglu_module.importlib, "import_module", raise_import_error)

    with pytest.raises(ModuleNotFoundError, match="cann_ops_nn"):
        _swiglu_group_converter().convert(model)

    assert model.experts is routed_experts
    assert isinstance(model.experts, GroupedExperts)
    assert not isinstance(model.experts, gmm_module.NpuGroupedExperts)
    assert not hasattr(model.experts, _EXPERT_ACTIVATION_FN_ATTR)
    assert model.experts.w1 is routed_weights[0]
    assert model.experts.w2 is routed_weights[1]
    assert model.experts.w3 is routed_weights[2]
    assert model.moe.shared_experts is shared_experts
    assert isinstance(model.moe.shared_experts, FeedForward)
    assert not isinstance(model.moe.shared_experts, common_moe.NpuSharedExperts)
    assert model.moe.shared_experts.w1 is shared_linears[0]
    assert model.moe.shared_experts.w2 is shared_linears[1]
    assert model.moe.shared_experts.w3 is shared_linears[2]
    assert tuple(model.state_dict()) == state_dict_keys


def test_converter_does_not_import_cann_ops_without_compatible_targets(monkeypatch):
    model = nn.Sequential(nn.Linear(2, 2))

    def fail_if_imported(_name):
        raise AssertionError("models without compatible targets must not import cann_ops_nn")

    monkeypatch.setattr(swiglu_module.importlib, "import_module", fail_if_imported)

    _swiglu_group_converter().convert(model)


@pytest.mark.parametrize(
    "converter_order",
    [
        ("npu_gmm", "npu_swiglu_group"),
        ("npu_swiglu_group", "npu_gmm"),
    ],
    ids=("gmm-then-swiglu-group", "swiglu-group-then-gmm"),
)
def test_npu_gmm_and_swiglu_group_are_order_independent(monkeypatch, converter_order):
    _allow_cann_ops_import(monkeypatch)
    model = _FeedForwardContainer()
    shared_experts = model.moe.shared_experts
    model_spec = SimpleNamespace(
        name="deepseek_v4",
        state_dict_adapter=_IdentityStateDictAdapter,
    )

    def fused_activation(h, swiglu_limit=None, routed_scores=None):
        return h

    monkeypatch.setattr(swiglu_module, "swiglu_group_activation", fused_activation)
    # Resolve after test_registry reloads converter modules during the same test session.
    converter_configs = {
        "npu_gmm": gmm_module.GMMModelConfig,
        "npu_swiglu_group": swiglu_module.SwigluGroupModelConfig,
    }
    first_config = converter_configs.get(converter_order[0])
    second_config = converter_configs.get(converter_order[1])
    assert first_config is not None
    assert second_config is not None

    _apply_model_config(model, model_spec, first_config)
    converted_experts = model.experts
    _apply_model_config(model, model_spec, second_config)

    assert model.experts is converted_experts
    assert isinstance(model.experts, gmm_module.NpuGroupedExperts)
    assert getattr(model.experts, _EXPERT_ACTIVATION_FN_ATTR) is fused_activation
    assert model.moe.shared_experts is shared_experts
    assert isinstance(shared_experts, common_moe.NpuSharedExperts)
    assert getattr(shared_experts, _EXPERT_ACTIVATION_FN_ATTR) is fused_activation
    assert (
        model_spec.state_dict_adapter.add_state_dict_updater(
            gmm_module.GMMStateDictUpdater
        )
        is False
    )


def test_npu_swiglu_group_converts_shared_expert_without_npu_gmm(
    monkeypatch,
    prepare_shared_expert_for_conversion,
):
    _allow_cann_ops_import(monkeypatch)
    model = _SharedExpertContainer()
    dense = model.feed_forward
    shared = model.moe.shared_experts
    state = prepare_shared_expert_for_conversion(model, shared)
    model_spec = SimpleNamespace(
        name="deepseek_v4",
        state_dict_adapter=_IdentityStateDictAdapter,
    )
    operator_calls = []

    def fused_activation(x, swiglu_limit=None, routed_scores=None):
        operator_calls.append((x, swiglu_limit, routed_scores))
        return x[..., : x.shape[-1] // 2]

    monkeypatch.setattr(swiglu_module, "swiglu_group_activation", fused_activation)

    _apply_model_config(model, model_spec, swiglu_module.SwigluGroupModelConfig)
    converted_shared = model.moe.shared_experts
    x = torch.ones(1, 2)
    expected_packed = torch.cat((state.linears[0](x), state.linears[2](x)), dim=-1)
    converted_shared(x)

    assert model.feed_forward is dense
    assert converted_shared is shared
    assert isinstance(converted_shared, common_moe.NpuSharedExperts)
    assert getattr(converted_shared, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation
    assert (converted_shared.w1, converted_shared.w2, converted_shared.w3) == state.linears
    assert converted_shared.conversion_marker is state.marker
    assert not converted_shared.training
    assert state.hook_calls == [True]
    assert tuple(model.state_dict()) == state.state_dict_keys
    assert torch.equal(operator_calls[0][0], expected_packed)
    assert operator_calls[0][0].is_contiguous()
    assert operator_calls[0][1:] == (2.0, None)
    assert model_spec.state_dict_adapter is _IdentityStateDictAdapter


def test_swiglu_group_does_not_own_routed_expert_execution():
    assert not hasattr(swiglu_module, "SwigluGroupExperts")
    assert not hasattr(swiglu_module, "_run_experts_for_loop")
    assert not hasattr(swiglu_module, "_run_experts_grouped_mm")


def test_swiglu_group_converts_raw_routed_and_shared_experts(monkeypatch):
    _allow_cann_ops_import(monkeypatch)
    model = _FeedForwardContainer()
    shared_experts = model.moe.shared_experts

    _swiglu_group_converter().convert(model)

    assert isinstance(model.experts, gmm_module.NpuGroupedExperts)
    assert getattr(model.experts, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation
    assert model.moe.shared_experts is shared_experts
    assert isinstance(shared_experts, common_moe.NpuSharedExperts)
    assert getattr(shared_experts, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation


def test_swiglu_group_converts_raw_routed_experts_without_shared_experts(monkeypatch):
    _allow_cann_ops_import(monkeypatch)
    model = nn.Module()
    model.experts = _grouped_experts()

    _swiglu_group_converter().convert(model)

    assert isinstance(model.experts, gmm_module.NpuGroupedExperts)
    assert model.experts.w13 is not None
    assert model.experts.w1 is None
    assert model.experts.w3 is None
    assert getattr(model.experts, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation


def test_swiglu_group_converts_mixed_raw_and_npu_routed_experts(monkeypatch):
    _allow_cann_ops_import(monkeypatch)
    converted_model = nn.Module()
    converted_model.experts = _grouped_experts()
    gmm_module.NpuGroupedExpertConverter(SimpleNamespace(name="deepseek_v4")).convert(
        converted_model
    )
    model = _SharedExpertContainer()
    model.raw_experts = _grouped_experts()
    model.npu_experts = converted_model.experts
    raw_experts = model.raw_experts
    npu_experts = model.npu_experts

    _swiglu_group_converter().convert(model)

    assert model.raw_experts is not raw_experts
    assert isinstance(model.raw_experts, gmm_module.NpuGroupedExperts)
    assert model.npu_experts is npu_experts
    assert getattr(model.raw_experts, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation
    assert getattr(model.npu_experts, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation


def test_swiglu_group_then_moe_dispatch_preserves_shared_activation(monkeypatch):
    _allow_cann_ops_import(monkeypatch)
    model = _SharedExpertContainer()

    _swiglu_group_converter().convert(model)
    shared = model.moe.shared_experts
    _moe_dispatch_converter().convert(model)

    assert model.moe.shared_experts is shared
    assert isinstance(shared, common_moe.NpuSharedExperts)
    assert getattr(shared, _EXPERT_ACTIVATION_FN_ATTR) is swiglu_module.swiglu_group_activation


def test_converted_shared_expert_uses_registered_operator_autograd(monkeypatch):
    _allow_cann_ops_import(monkeypatch)
    model = _SharedExpertContainer()
    shared = model.moe.shared_experts
    for linear in (shared.w1, shared.w2, shared.w3):
        with torch.no_grad():
            linear.weight.copy_(torch.eye(2))

    _install_fake_ops(monkeypatch, _slice_swiglu_output)
    _swiglu_group_converter().convert(model)

    x = torch.ones(1, 2, requires_grad=True)
    model.moe.shared_experts(x).sum().backward()

    assert x.grad is not None
    assert shared.w1.weight.grad is not None
    assert shared.w2.weight.grad is not None
    assert shared.w3.weight.grad is not None


def test_npu_moe_dispatch_and_gmm_keep_native_activations():
    model = _FeedForwardContainer()
    shared = model.moe.shared_experts

    _moe_dispatch_converter().convert(model)
    gmm_module.NpuGroupedExpertConverter(SimpleNamespace(name="deepseek_v4")).convert(model)

    native_activation = getattr(gmm_module, _EXPERT_ACTIVATION_ATTR)
    assert getattr(model.experts, _EXPERT_ACTIVATION_FN_ATTR) is native_activation
    assert model.moe.shared_experts is shared
    assert isinstance(shared, common_moe.NpuSharedExperts)
    assert getattr(shared, _EXPERT_ACTIVATION_FN_ATTR) is common_moe.native_shared_expert_activation
