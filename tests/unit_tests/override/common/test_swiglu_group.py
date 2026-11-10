# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real typed recipe overrides compose at disjoint activation/dispatcher leaves.

CANN packages and operator registration remain real. The last test substitutes
only the registered device primitive on CPU to check the built activation's
packing, clamp argument, output and gradient boundary; it is not NPU validation.
"""

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import ConfigManager, apply_overrides
from torchtitan.models.common.activation import SiTUGLU, SwiGLU

from torchtitan_npu.override.common.swiglu_group import AscSwiGLU
from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher
from torchtitan_npu.patches.torchtitan.models.common.moe import ClampedSwiGLU

_SWIGLU = "torchtitan_npu.override.common.swiglu_group."
_DISPATCHERS = {
    "standard": ("torchtitan_npu.override.common.token_dispatcher.asc", AscAllToAllTokenDispatcher.Config),
}
_RECIPES = [
    ("torchtitan_npu.models.deepseek_v4.config_registry", "deepseek_v4_debugmodel"),
    ("torchtitan_npu.models.deepseek_v4_1.config_registry", "deepseek_v4_1_debugmodel_text"),
]


def _trainer_config(module, recipe, tmp_path):
    config = ConfigManager().parse_args(["--module", module, "--config", recipe, "--hf-assets-path", str(tmp_path)])
    config.model.update_from_config(config=config)
    return config


def _moe_layers(config):
    return [(index, layer) for index, layer in enumerate(config.model.layers) if layer.moe is not None]


def _replacement_fqns(replacements):
    return {line.split(": ", 1)[1].split(" ", 1)[0] for line in replacements}


@pytest.mark.parametrize(("module", "recipe"), _RECIPES, ids=["v4", "v41"])
@pytest.mark.parametrize("dispatcher", ["standard"])
@pytest.mark.parametrize("dispatcher_first", [False, True])
def test_activation_and_dispatcher_overrides_compose_without_replacing_expert_owners(
    tmp_path, module, recipe, dispatcher, dispatcher_first
):
    config = _trainer_config(module, recipe, tmp_path)
    layers = _moe_layers(config)
    assert layers
    originals = [
        (
            layer.moe,
            layer.moe.routed_experts,
            layer.moe.shared_experts,
            layer.moe.routed_experts.w13,
            layer.moe.routed_experts.w2,
            layer.moe.shared_experts.w13,
            layer.moe.shared_experts.w2,
            layer.moe.routed_experts.activation_fn.swiglu_limit,
            getattr(layer.moe.shared_experts.activation_fn, "swiglu_limit", 0.0),
            layer.moe.routed_experts.token_dispatcher.absorb_router_scores,
            layer.attention,
        )
        for _, layer in layers
    ]
    assert any(isinstance(layer.moe.routed_experts.activation_fn, ClampedSwiGLU.Config) for _, layer in layers)
    dispatcher_target, dispatcher_config = _DISPATCHERS[dispatcher]
    imports = [_SWIGLU + "asc", _SWIGLU + "asc_shared_experts"]
    imports.insert(0 if dispatcher_first else len(imports), dispatcher_target)
    config.override.imports = imports

    replacements = apply_overrides(config.override, config)
    config.__post_init__()

    expected_fqns = set()
    for (index, layer), original in zip(layers, originals, strict=True):
        (
            moe,
            routed,
            shared,
            routed_w13,
            routed_w2,
            shared_w13,
            shared_w2,
            limit,
            shared_limit,
            absorb,
            attention,
        ) = original
        assert layer.moe is moe
        assert layer.moe.routed_experts is routed
        assert layer.moe.shared_experts is shared
        assert routed.w13 is routed_w13 and routed.w2 is routed_w2
        assert shared.w13 is shared_w13 and shared.w2 is shared_w2
        assert layer.attention is attention
        assert type(routed.activation_fn) is AscSwiGLU.Config
        assert type(shared.activation_fn) is AscSwiGLU.Config
        assert routed.activation_fn.swiglu_limit == limit
        assert shared.activation_fn.swiglu_limit == shared_limit
        assert type(routed.token_dispatcher) is dispatcher_config
        assert routed.token_dispatcher.absorb_router_scores is absorb
        prefix = f"model.layers.{index}.moe"
        expected_fqns.update(
            {
                prefix + ".routed_experts.activation_fn",
                prefix + ".shared_experts.activation_fn",
                prefix + ".routed_experts.token_dispatcher",
            }
        )
    assert len(replacements) == len(expected_fqns)
    assert _replacement_fqns(replacements) == expected_fqns


@pytest.mark.parametrize("selected", ["routed_experts", "shared_experts"])
def test_only_requested_swiglu_leaf_changes_and_plain_swiglu_has_no_clamp(tmp_path, selected):
    config = _trainer_config(*_RECIPES[0], tmp_path)
    first_moe = _moe_layers(config)[0][1].moe
    first_moe.routed_experts.activation_fn = SwiGLU.Config()
    first_moe.shared_experts.activation_fn = ClampedSwiGLU.Config(swiglu_limit=0.75)
    routed, shared = first_moe.routed_experts, first_moe.shared_experts
    original_routed, original_shared = routed.activation_fn, shared.activation_fn
    config.override.imports = [_SWIGLU + ("asc" if selected == "routed_experts" else "asc_shared_experts")]

    replacements = apply_overrides(config.override, config)

    assert all(fqn.endswith(f".moe.{selected}.activation_fn") for fqn in _replacement_fqns(replacements))
    if selected == "routed_experts":
        assert type(routed.activation_fn) is AscSwiGLU.Config
        assert routed.activation_fn.swiglu_limit == 0.0
        assert shared.activation_fn is original_shared
    else:
        assert type(shared.activation_fn) is AscSwiGLU.Config
        assert shared.activation_fn.swiglu_limit == 0.75
        assert routed.activation_fn is original_routed


@pytest.mark.parametrize(("module", "recipe"), _RECIPES, ids=["v4", "v41"])
def test_situglu_activations_are_preserved_while_dispatcher_is_selected(tmp_path, module, recipe):
    config = _trainer_config(module, recipe, tmp_path)
    originals = []
    for _, layer in _moe_layers(config):
        routed = SiTUGLU.Config(beta=0.5, linear_beta=1.5)
        shared = SiTUGLU.Config(beta=1.25)
        layer.moe.routed_experts.activation_fn = routed
        layer.moe.shared_experts.activation_fn = shared
        originals.append((routed, shared))
    config.override.imports = [_SWIGLU + "asc", _SWIGLU + "asc_shared_experts", _DISPATCHERS["standard"][0]]

    replacements = apply_overrides(config.override, config)

    assert replacements
    assert all(fqn.endswith(".routed_experts.token_dispatcher") for fqn in _replacement_fqns(replacements))
    for (_, layer), (routed, shared) in zip(_moe_layers(config), originals, strict=True):
        assert layer.moe.routed_experts.activation_fn is routed
        assert layer.moe.shared_experts.activation_fn is shared
        assert (routed.beta, routed.linear_beta, shared.beta) == (0.5, 1.5, 1.25)


@pytest.mark.parametrize("limit", [0.0, 0.75])
def test_recipe_selected_built_activation_reaches_registered_cann_boundary(monkeypatch, tmp_path, limit):
    config = _trainer_config(*_RECIPES[0], tmp_path)
    first_moe = _moe_layers(config)[0][1].moe
    first_moe.routed_experts.activation_fn = ClampedSwiGLU.Config(swiglu_limit=limit)
    config.override.imports = [_SWIGLU + "asc"]
    apply_overrides(config.override, config)
    activation = first_moe.routed_experts.activation_fn.build()
    assert type(activation) is AscSwiGLU
    # The real override above loaded cann_ops_nn.ops and registered this operator.
    operation = torch.ops.cann_ops_nn.swiglu_group
    assert operation.default._schema.name == "cann_ops_nn::swiglu_group"
    calls = []

    def cpu_device_primitive(h, *, weight, group_index, clamp_limit):
        calls.append((h, weight, group_index, clamp_limit))
        gate, up = h.chunk(2, dim=-1)
        if clamp_limit > 0:
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(min=-clamp_limit, max=clamp_limit)
        return F.silu(gate) * up

    monkeypatch.setattr(operation, "default", cpu_device_primitive)
    gate = torch.tensor([[-2.0, 1.0, 0.5], [0.25, 3.0, -1.0]], requires_grad=True)
    up = torch.tensor([[2.0, -3.0, 0.25], [0.5, 1.0, -2.0]], requires_grad=True)
    ref_gate, ref_up = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
    clamped_gate = ref_gate.clamp(max=limit) if limit > 0 else ref_gate
    clamped_up = ref_up.clamp(min=-limit, max=limit) if limit > 0 else ref_up
    expected = F.silu(clamped_gate) * clamped_up

    actual = activation(gate, up, offsets=torch.tensor([2]))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(gate.grad, ref_gate.grad, rtol=0, atol=0)
    torch.testing.assert_close(up.grad, ref_up.grad, rtol=0, atol=0)
    assert len(calls) == 1
    packed, weight, group_index, clamp_limit = calls[0]
    torch.testing.assert_close(packed, torch.cat((gate, up), dim=-1))
    assert packed.is_contiguous()
    assert weight is None and group_index is None
    assert clamp_limit == (limit if limit > 0 else -1.0)
