# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real typed recipe overrides compose at disjoint activation/dispatcher leaves.

These are config-level checks: ``_ensure_cann_ops_loaded`` is faked so the CPU
UT does not depend on the CANN JIT build. The real ``cann_ops_nn.swiglu_group``
registration boundary is covered by
``tests/smoke_tests/ops/test_swiglu_group_cann_boundary.py``.
"""

import pytest
from torchtitan.config import apply_overrides
from torchtitan.config.loader import ConfigLoader
from torchtitan.models.common.activation import SiTUGLU, SwiGLU
from torchtitan.models.common.moe import MoE

from torchtitan_npu.override.common.swiglu_group import AscSwiGLU
from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher
from torchtitan_npu.patches.torchtitan.models.common.moe import LimitedSwiGLU

_SWIGLU = "torchtitan_npu.override.common.swiglu_group."
_DISPATCHERS = {
    "standard": ("torchtitan_npu.override.common.token_dispatcher.asc", AscAllToAllTokenDispatcher.Config),
}
_RECIPES = [
    ("torchtitan_npu_recipes.models.deepseek_v4", "deepseek_v4_debugmodel"),
    ("torchtitan_npu_recipes.models.deepseek_v4_1", "deepseek_v4_1_debugmodel_text"),
]


def _trainer_config(module, recipe, tmp_path):
    del tmp_path
    return ConfigLoader._load_config(module, recipe)


def _moe_layers(config):
    return [
        (index, layer) for index, layer in enumerate(config.model.layers) if isinstance(layer.feed_forward, MoE.Config)
    ]


def _replacement_fqns(replacements):
    return {line.split(": ", 1)[1].split(" ", 1)[0] for line in replacements}


@pytest.fixture
def cann_free_swiglu(monkeypatch):
    """Keep the config-only override tests independent of the CANN JIT build."""
    from torchtitan_npu.override.common import swiglu_group

    monkeypatch.setattr(swiglu_group, "_ensure_cann_ops_loaded", lambda: None)


@pytest.mark.parametrize(("module", "recipe"), _RECIPES, ids=["v4", "v41"])
@pytest.mark.parametrize("dispatcher", ["standard"])
@pytest.mark.parametrize("dispatcher_first", [False, True])
def test_activation_and_dispatcher_overrides_compose_without_replacing_expert_owners(
    tmp_path, module, recipe, dispatcher, dispatcher_first, cann_free_swiglu
):
    config = _trainer_config(module, recipe, tmp_path)
    layers = _moe_layers(config)
    assert layers
    originals = [
        (
            layer.feed_forward,
            layer.feed_forward.routed_experts,
            layer.feed_forward.shared_experts,
            layer.feed_forward.routed_experts.w13,
            layer.feed_forward.routed_experts.w2,
            layer.feed_forward.shared_experts.w13,
            layer.feed_forward.shared_experts.w2,
            layer.feed_forward.routed_experts.activation_fn.swiglu_limit,
            getattr(layer.feed_forward.shared_experts.activation_fn, "swiglu_limit", 0.0),
            layer.feed_forward.routed_experts.token_dispatcher.absorb_router_scores,
            layer.attention,
        )
        for _, layer in layers
    ]
    assert any(isinstance(layer.feed_forward.routed_experts.activation_fn, LimitedSwiGLU.Config) for _, layer in layers)
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
        assert layer.feed_forward is moe
        assert layer.feed_forward.routed_experts is routed
        assert layer.feed_forward.shared_experts is shared
        assert routed.w13 is routed_w13 and routed.w2 is routed_w2
        assert shared.w13 is shared_w13 and shared.w2 is shared_w2
        assert layer.attention is attention
        assert type(routed.activation_fn) is AscSwiGLU.Config
        assert type(shared.activation_fn) is AscSwiGLU.Config
        assert routed.activation_fn.swiglu_limit == limit
        assert shared.activation_fn.swiglu_limit == shared_limit
        assert type(routed.token_dispatcher) is dispatcher_config
        assert routed.token_dispatcher.absorb_router_scores is absorb
        prefix = f"model.layers.{index}.feed_forward"
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
def test_only_requested_swiglu_leaf_changes_and_plain_swiglu_has_no_clamp(tmp_path, selected, cann_free_swiglu):
    config = _trainer_config(*_RECIPES[0], tmp_path)
    first_moe = _moe_layers(config)[0][1].feed_forward
    first_moe.routed_experts.activation_fn = SwiGLU.Config()
    first_moe.shared_experts.activation_fn = LimitedSwiGLU.Config(swiglu_limit=0.75)
    routed, shared = first_moe.routed_experts, first_moe.shared_experts
    original_routed, original_shared = routed.activation_fn, shared.activation_fn
    config.override.imports = [_SWIGLU + ("asc" if selected == "routed_experts" else "asc_shared_experts")]

    replacements = apply_overrides(config.override, config)

    assert all(fqn.endswith(f".feed_forward.{selected}.activation_fn") for fqn in _replacement_fqns(replacements))
    if selected == "routed_experts":
        assert type(routed.activation_fn) is AscSwiGLU.Config
        assert routed.activation_fn.swiglu_limit == 0.0
        assert shared.activation_fn is original_shared
    else:
        assert type(shared.activation_fn) is AscSwiGLU.Config
        assert shared.activation_fn.swiglu_limit == 0.75
        assert routed.activation_fn is original_routed


@pytest.mark.parametrize(("module", "recipe"), _RECIPES, ids=["v4", "v41"])
def test_situglu_activations_are_preserved_while_dispatcher_is_selected(tmp_path, module, recipe, cann_free_swiglu):
    config = _trainer_config(module, recipe, tmp_path)
    originals = []
    for _, layer in _moe_layers(config):
        routed = SiTUGLU.Config(beta=0.5, linear_beta=1.5)
        shared = SiTUGLU.Config(beta=1.25)
        layer.feed_forward.routed_experts.activation_fn = routed
        layer.feed_forward.shared_experts.activation_fn = shared
        originals.append((routed, shared))
    config.override.imports = [_SWIGLU + "asc", _SWIGLU + "asc_shared_experts", _DISPATCHERS["standard"][0]]

    replacements = apply_overrides(config.override, config)

    assert replacements
    assert all(fqn.endswith(".routed_experts.token_dispatcher") for fqn in _replacement_fqns(replacements))
    for (_, layer), (routed, shared) in zip(_moe_layers(config), originals, strict=True):
        assert layer.feed_forward.routed_experts.activation_fn is routed
        assert layer.feed_forward.shared_experts.activation_fn is shared
        assert (routed.beta, routed.linear_beta, shared.beta) == (0.5, 1.5, 1.25)
