# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from unittest.mock import Mock, patch

import torch.nn as nn
from torch.distributed.tensor.parallel import parallelize_module

from torchtitan_npu.converters import (
    ModelCustomConfig,
    ParallelizePlanUpdater,
    StateDictUpdater,
    registry,
)
from torchtitan_npu.converters.framework.model_custom_config_converter import (
    ModelCustomConfigConverter,
)
from torchtitan_npu.converters.framework.parallelize_plan_update_wrapper import (
    apply_parallelize_plan_update,
)
from torchtitan_npu.converters.framework.state_dict_update_wrapper import (
    apply_state_dict_update,
    get_state_dict_adapter_wrapper,
)
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter


@contextmanager
def isolated_model_config_registry():
    test_registry = registry()
    old_model_configs = dict(test_registry._model_configs)
    old_converter_classes = dict(test_registry._converter_classes)
    test_registry._model_configs = {}
    test_registry._converter_classes = {}
    try:
        yield test_registry
    finally:
        test_registry._model_configs = old_model_configs
        test_registry._converter_classes = old_converter_classes


class TestApplyParallelizePlanUpdater:
    @classmethod
    def test_apply_parallelize_plan_updater(cls):
        call_order = []

        class CustomParallelizePlanUpdater(ParallelizePlanUpdater):
            @classmethod
            def update(cls, layer_plan):
                call_order.append("updater_update")
                layer_plan["modified_by_updater"] = True
                return layer_plan

        parallelize_fn_called = []

        def parallelize_fn(module, device_mesh, parallelize_plan):
            parallelize_fn_called.append(True)
            parallelize_module(module, device_mesh, parallelize_plan)
            assert parallelize_plan.get("modified_by_updater") is True, (
                "updater should have been called before parallelize_fn"
            )
            return module

        apply_parallelize_plan_update(CustomParallelizePlanUpdater)

        with patch(
            "torchtitan_npu.converters.framework.parallelize_plan_update_wrapper._torch_parallelize_module"
        ) as mock_parallelize_module:
            mock_parallelize_module.return_value = nn.Linear(10, 10)

            parallelize_fn(
                module=nn.Linear(10, 10),
                device_mesh=None,
                parallelize_plan={"test": Mock()},
            )

            assert "updater_update" in call_order, "ParallelizePlanUpdater.update should be called"
            assert mock_parallelize_module.called, "parallelize_module should be called"


class TestApplyStateDictUpdateIntegration:
    @classmethod
    def test_wrapper_owns_updater_registration(cls):
        class MockOriginalAdapter:
            pass

        class CustomStateDictUpdater(StateDictUpdater):
            pass

        adapter_wrapper = get_state_dict_adapter_wrapper(MockOriginalAdapter)

        assert adapter_wrapper.add_state_dict_updater(CustomStateDictUpdater) is True
        assert adapter_wrapper.add_state_dict_updater(CustomStateDictUpdater) is False

    @classmethod
    def test_apply_state_dict_updater(cls):
        call_order = []

        class MockOriginalAdapter:
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("original_to_hf")
                state_dict["original_to_hf_applied"] = True
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("original_from_hf")
                state_dict["original_from_hf_applied"] = True
                return state_dict

        class CustomStateDictUpdater(StateDictUpdater):
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("updater_to_hf")
                state_dict["updater_to_hf_applied"] = True
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("updater_from_hf")
                state_dict["updater_from_hf_applied"] = True
                return state_dict

        train_spec = Mock()
        train_spec.state_dict_adapter = MockOriginalAdapter

        apply_state_dict_update(CustomStateDictUpdater, train_spec)

        assert train_spec.state_dict_adapter is not MockOriginalAdapter, "state_dict_adapter should be wrapped"

        state_dict = {}
        adapter = train_spec.state_dict_adapter()
        result_to_hf = adapter.to_hf(state_dict)

        assert "updater_to_hf" in call_order, "updater.to_hf should be called"
        assert "original_to_hf" in call_order, "original adapter.to_hf should be called"
        updater_to_hf_idx = call_order.index("updater_to_hf")
        original_to_hf_idx = call_order.index("original_to_hf")
        assert updater_to_hf_idx < original_to_hf_idx, "updater.to_hf should be called BEFORE original adapter.to_hf"

        assert result_to_hf.get("updater_to_hf_applied") is True
        assert result_to_hf.get("original_to_hf_applied") is True

        call_order.clear()
        state_dict = {}
        result_from_hf = adapter.from_hf(state_dict)

        assert "updater_from_hf" in call_order, "updater.from_hf should be called"
        assert "original_from_hf" in call_order, "original adapter.from_hf should be called"
        updater_from_hf_idx = call_order.index("updater_from_hf")
        original_from_hf_idx = call_order.index("original_from_hf")
        assert original_from_hf_idx < updater_from_hf_idx, (
            "updater.from_hf should be called AFTER original adapter.from_hf"
        )

        assert result_from_hf.get("updater_from_hf_applied") is True
        assert result_from_hf.get("original_from_hf_applied") is True

    @classmethod
    def test_duplicate_updater_type_is_applied_once(cls):
        call_order = []

        class CustomStateDictUpdater(StateDictUpdater):
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("updater_to_hf")
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("updater_from_hf")
                return state_dict

        class MockOriginalAdapter:
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("original_to_hf")
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("original_from_hf")
                return state_dict

        train_spec = Mock()
        train_spec.state_dict_adapter = MockOriginalAdapter

        apply_state_dict_update(CustomStateDictUpdater, train_spec)
        apply_state_dict_update(CustomStateDictUpdater, train_spec)

        adapter = train_spec.state_dict_adapter()
        adapter.to_hf({})
        assert call_order == ["updater_to_hf", "original_to_hf"]

        call_order.clear()
        adapter.from_hf({})
        assert call_order == ["original_from_hf", "updater_from_hf"]


class TestModelCustomConfigConverter:
    @classmethod
    def test_state_dict_updater_predicate_uses_converted_model(cls):
        class CapabilityConverter(ModelCustomConverter):
            def convert(self, model):
                model.has_state_dict_update_capability = model.enable_capability

        class ConditionalStateDictUpdater(StateDictUpdater):
            @classmethod
            def to_hf(cls, state_dict):
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                return state_dict

        class ConditionalConfig(ModelCustomConfig):
            name = "conditional_state_dict_update"
            model_converter = CapabilityConverter
            state_dict_updater = ConditionalStateDictUpdater
            state_dict_updater_predicate = staticmethod(
                lambda model: model.has_state_dict_update_capability
            )

        class ConditionalConfigConverter(ModelCustomConfigConverter):
            _model_config = ConditionalConfig

        converter = ConditionalConfigConverter.__new__(ConditionalConfigConverter)
        converter.model_spec = Mock()

        with patch(
            "torchtitan_npu.converters.framework.model_custom_config_converter.apply_state_dict_update"
        ) as mock_apply_state_dict_update:
            disabled_model = nn.Module()
            disabled_model.enable_capability = False
            converter.convert(disabled_model)
            mock_apply_state_dict_update.assert_not_called()

            enabled_model = nn.Module()
            enabled_model.enable_capability = True
            converter.convert(enabled_model)
            mock_apply_state_dict_update.assert_called_once_with(
                ConditionalStateDictUpdater,
                converter.model_spec,
            )


class TestRegisterModelConverter:
    @classmethod
    def test_register_model_converter_adds_to_model_configs(cls):
        with isolated_model_config_registry() as test_registry:

            @test_registry.register("test_model_config")
            class TestModelConfig(ModelCustomConfig):
                pass

            assert "test_model_config" in test_registry._model_configs
            assert test_registry._model_configs["test_model_config"] is TestModelConfig
            assert test_registry.get("test_model_config") is TestModelConfig

    @classmethod
    def test_set_name(cls):
        with isolated_model_config_registry():

            @registry().register("custom_model_name")
            class CustomModelConfig(ModelCustomConfig):
                pass

            assert CustomModelConfig.name == "custom_model_name"

    @classmethod
    def test_create_converter_class(cls):
        with isolated_model_config_registry() as test_registry:

            @test_registry.register("test_converter_creation")
            class TestConfig(ModelCustomConfig):
                pass

            registered_cls = test_registry._converter_classes["test_converter_creation"]

            assert issubclass(registered_cls, ModelCustomConfigConverter)
            assert hasattr(registered_cls, "_model_config")
            assert registered_cls._model_config is TestConfig
            assert registered_cls.Config._owner is registered_cls

    @classmethod
    def test_multiple_updaters_in_order(cls):
        call_order = []

        class FirstUpdater(StateDictUpdater):
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("first")
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("first_from")
                return state_dict

        class SecondUpdater(StateDictUpdater):
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("second")
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("second_from")
                return state_dict

        class MockAdapter:
            @classmethod
            def to_hf(cls, state_dict):
                call_order.append("base")
                return state_dict

            @classmethod
            def from_hf(cls, state_dict):
                call_order.append("base_from")
                return state_dict

        train_spec = Mock()
        train_spec.state_dict_adapter = MockAdapter

        apply_state_dict_update(FirstUpdater, train_spec)
        apply_state_dict_update(SecondUpdater, train_spec)

        call_order.clear()
        adapter = train_spec.state_dict_adapter()
        adapter.to_hf({})

        assert call_order == ["first", "second", "base"]

        call_order.clear()
        adapter.from_hf({})

        assert call_order == ["base_from", "first_from", "second_from"]
