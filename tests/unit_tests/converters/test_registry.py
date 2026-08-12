# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch.nn as nn

from torchtitan_npu.converters import registry
from torchtitan_npu.converters.kernels import gmm as gmm_module


@contextmanager
def isolated_registry():
    test_registry = registry()
    old_model_configs = dict(test_registry._model_configs)
    old_converter_classes = dict(test_registry._converter_classes)
    try:
        yield test_registry
    finally:
        test_registry._model_configs = old_model_configs
        test_registry._converter_classes = old_converter_classes


class DummyModelCustomConfig:
    name: str = "dummy"


def _run_register_case(register_name):
    calls = []
    with isolated_registry() as test_registry:
        original_register = test_registry._register_as_model_converter

        def mock_register(name, config):
            calls.append((name, config))

        test_registry._register_as_model_converter = mock_register

        decorated_config = test_registry.register(register_name)(DummyModelCustomConfig)
        config = test_registry.get(register_name)

        test_registry._register_as_model_converter = original_register

        return decorated_config, config, calls


def test_registry_is_singleton():
    registry1 = registry()
    registry2 = registry()

    assert registry1 is registry2


def test_register_sets_name_and_stores_config():
    decorated_config, config, calls = _run_register_case("unit_dummy")

    assert decorated_config is DummyModelCustomConfig
    assert config is not None
    assert config.name == "unit_dummy"
    assert len(calls) == 1
    assert calls[0][0] == "unit_dummy"
    assert calls[0][1] is DummyModelCustomConfig


def test_get_returns_none_for_unknown_config():
    assert registry().get("definitely_missing_config") is None


def test_core_converter_registrations_exist():
    import importlib

    for module_name in (
        "torchtitan_npu.converters.kernels.dsa",
        "torchtitan_npu.converters.kernels.gmm",
        "torchtitan_npu.converters.kernels.swiglu_group",
        "torchtitan_npu.converters.kernels.rms_norm",
        "torchtitan_npu.converters.kernels.rope",
        "torchtitan_npu.converters.kernels.inplace_partial_rope",
        "torchtitan_npu.converters.kernels.moe_dispatch",
        "torchtitan_npu.converters.features.vlm",
    ):
        module = importlib.import_module(module_name)
        importlib.reload(module)

    expected_model_converter_names = [
        "npu_dsa",
        "npu_gmm",
        "npu_swiglu_group",
        "npu_rms_norm",
        "npu_rope",
        "npu_rope_inplace_partial",
        "npu_moe_dispatch",
        "npu_moe_reduce_fused_tilelang",
        "npu_vlm",
    ]
    for name in expected_model_converter_names:
        config = registry().get(name)
        assert config is not None, f"{name} should be registered"
        assert config.name == name

    swiglu_config = registry().get("npu_swiglu_group")
    assert swiglu_config is not None
    assert swiglu_config.state_dict_updater is gmm_module.GMMStateDictUpdater
    state_dict_updater_predicate = swiglu_config.state_dict_updater_predicate
    assert callable(state_dict_updater_predicate)
    assert state_dict_updater_predicate(nn.Module()) is False

    model_with_npu_grouped_experts = nn.Module()
    npu_grouped_experts = gmm_module.NpuGroupedExperts.__new__(
        gmm_module.NpuGroupedExperts
    )
    nn.Module.__init__(npu_grouped_experts)
    model_with_npu_grouped_experts.add_module("experts", npu_grouped_experts)
    assert state_dict_updater_predicate(model_with_npu_grouped_experts) is True
    assert registry().get("npu_gmm_swiglu") is None
