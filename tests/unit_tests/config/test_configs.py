# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
from dataclasses import fields

from torchtitan.config import ConfigManager
from torchtitan.config import TrainingConfig as UpstreamTrainingConfig
from torchtitan.trainer import Trainer

from torchtitan_npu.config import (
    ExtensionConfig,
    QuantizationExtensionConfig,
    TrainerConfig,
    TrainingExtensionConfig,
)
from torchtitan_npu.config import (
    TrainingConfig as NPUTrainingConfig,
)
from torchtitan_npu.distributed import utils as distributed_utils
from torchtitan_npu.extensions.trainer import TrainerEx


def test_from_trainer_config_wraps_training_extension():
    source = Trainer.Config(
        dump_folder="custom-output",
        training=UpstreamTrainingConfig(
            local_batch_size=3,
            seq_len=4096,
            steps=17,
        ),
    )

    adapted = TrainerConfig.from_trainer_config(source)

    assert isinstance(adapted, TrainerConfig)
    assert isinstance(adapted.extension, ExtensionConfig)
    assert isinstance(adapted.extension.quantization, QuantizationExtensionConfig)
    assert adapted.extension.quantization.enable_quantized_training is False
    assert adapted.extension.quantization.recipe == "mix"
    assert adapted.extension.quantization.enable_mxfp4_qat is False
    assert adapted.extension.quantization.dst_type_max == 0.0
    assert isinstance(adapted.training, NPUTrainingConfig)
    assert isinstance(adapted.training.extension, TrainingExtensionConfig)
    assert adapted.training.extension.allow_hf32 is True
    for config_field in fields(Trainer.Config):
        if config_field.name == "training":
            continue
        assert getattr(adapted, config_field.name) == getattr(source, config_field.name)
    for config_field in fields(UpstreamTrainingConfig):
        assert getattr(adapted.training, config_field.name) == getattr(source.training, config_field.name)


def test_config_manager_parses_training_extension_into_trainer_ex_config(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_test_config_registry"
    registry = types.ModuleType(module_name)

    def test_config() -> Trainer.Config:
        return Trainer.Config(
            hf_assets_path=str(tmp_path),
            training=UpstreamTrainingConfig(steps=23),
        )

    registry.test_config = test_config
    monkeypatch.setitem(sys.modules, module_name, registry)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--training.extension.no-allow-hf32",
        ]
    )

    assert isinstance(config, TrainerConfig)
    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.training, NPUTrainingConfig)
    assert config.training.extension.allow_hf32 is False
    assert config.training.steps == 23


def test_config_manager_parses_quantization_extension(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_quantization_config_registry"
    registry = types.ModuleType(module_name)

    def test_config() -> Trainer.Config:
        return Trainer.Config(hf_assets_path=str(tmp_path))

    registry.test_config = test_config
    monkeypatch.setitem(sys.modules, module_name, registry)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--extension.quantization.enable-quantized-training",
            "--extension.quantization.recipe",
            "all_block_fp8",
            "--extension.quantization.enable-mxfp4-qat",
            "--extension.quantization.dst-type-max",
            "7.0",
        ]
    )

    quantization = config.extension.quantization
    assert quantization.enable_quantized_training is True
    assert quantization.recipe == "all_block_fp8"
    assert quantization.enable_mxfp4_qat is True
    assert quantization.dst_type_max == 7.0


def test_trainer_config_build_applies_extension_before_parent(monkeypatch):
    events = []
    expected_result = object()

    def apply_runtime(allow_hf32):
        events.append(("apply", allow_hf32))

    def parent_build(config, **kwargs):
        events.append(("build", kwargs))
        return expected_result

    monkeypatch.setattr(distributed_utils, "set_allow_hf32", apply_runtime)
    monkeypatch.setattr(TrainerEx.Config, "build", parent_build)

    config = TrainerConfig(
        training=NPUTrainingConfig(
            extension=TrainingExtensionConfig(allow_hf32=False),
        )
    )
    result = config.build(example="value")

    assert result is expected_result
    assert events == [
        ("apply", False),
        ("build", {"example": "value"}),
    ]


def test_trainer_config_build_quantizes_after_cli_config(monkeypatch):
    events = []
    source_model_spec = object()
    quantized_model_spec = object()
    converter_module = types.ModuleType("interfaces.torchao_converter")

    def apply_quantization(model_spec, quantization_config, *, model_compile_enabled):
        events.append(
            (
                "quantize",
                model_spec,
                quantization_config.recipe,
                quantization_config.enable_mxfp4_qat,
                quantization_config.dst_type_max,
                model_compile_enabled,
            )
        )
        return quantized_model_spec

    def apply_runtime(allow_hf32):
        events.append(("apply", allow_hf32))

    def parent_build(config, **kwargs):
        events.append(("build", config.model_spec, kwargs))
        return config.model_spec

    converter_module.apply_quantization_converter = apply_quantization
    monkeypatch.setitem(sys.modules, "interfaces.torchao_converter", converter_module)
    monkeypatch.setattr(distributed_utils, "set_allow_hf32", apply_runtime)
    monkeypatch.setattr(Trainer.Config, "build", parent_build)

    config = TrainerConfig(
        model_spec=source_model_spec,
        extension=ExtensionConfig(
            quantization=QuantizationExtensionConfig(
                enable_quantized_training=True,
                recipe="all_block_fp8",
                enable_mxfp4_qat=True,
                dst_type_max=7.0,
            ),
        ),
    )
    result = config.build(example="value")

    assert result is quantized_model_spec
    assert events == [
        ("quantize", source_model_spec, "all_block_fp8", True, 7.0, False),
        ("apply", True),
        ("build", quantized_model_spec, {"example": "value"}),
    ]


def test_trainer_config_build_skips_quantization_when_disabled(monkeypatch):
    source_model_spec = object()
    converter_module = types.ModuleType("interfaces.torchao_converter")

    def unexpected_quantization(*args, **kwargs):
        raise AssertionError("quantization converter must remain disabled")

    def parent_build(config, **kwargs):
        return config.model_spec

    converter_module.apply_quantization_converter = unexpected_quantization
    monkeypatch.setitem(sys.modules, "interfaces.torchao_converter", converter_module)
    monkeypatch.setattr(distributed_utils, "set_allow_hf32", lambda _: None)
    monkeypatch.setattr(Trainer.Config, "build", parent_build)

    config = TrainerConfig(
        model_spec=source_model_spec,
        extension=ExtensionConfig(
            quantization=QuantizationExtensionConfig(
                enable_quantized_training=False,
            ),
        ),
    )

    assert config.build() is source_model_spec


def test_trainer_config_build_constructs_trainer(monkeypatch):
    built_configs = []

    def capture_config(_self, config):
        built_configs.append(config)

    monkeypatch.setattr(distributed_utils, "set_allow_hf32", lambda _: None)
    monkeypatch.setattr(Trainer, "__init__", capture_config)

    trainer = TrainerConfig().build()

    assert isinstance(trainer, Trainer)
    assert len(built_configs) == 1
    assert isinstance(built_configs[0], TrainerConfig)
