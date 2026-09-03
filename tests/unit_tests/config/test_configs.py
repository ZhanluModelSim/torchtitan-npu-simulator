# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, fields

import pytest
from torchtitan.config import ConfigManager
from torchtitan.config import TrainingConfig as UpstreamTrainingConfig
from torchtitan.trainer import Trainer

import torchtitan_npu.config as npu_config
import torchtitan_npu.extensions.trainer as trainer_module
from torchtitan_npu.config import (
    ExtensionConfig,
    MuonOptimizerProfile,
    OptimizerConfig,
    QuantizationExtensionConfig,
    TrainingExtensionConfig,
)
from torchtitan_npu.config import TrainingConfig as NPUTrainingConfig
from torchtitan_npu.config import manager as config_manager
from torchtitan_npu.config.converters import TrainerConfigConverter
from torchtitan_npu.distributed import utils as distributed_utils
from torchtitan_npu.extensions.trainer import TrainerEx


def _install_config_registry(
    monkeypatch,
    module_name: str,
    factory: Callable[[], Trainer.Config],
) -> None:
    registry = types.ModuleType(module_name)
    registry.test_config = factory
    monkeypatch.setitem(sys.modules, module_name, registry)


def test_config_manager_adapts_standard_component_configs_without_changing_values(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_standard_config_registry"
    source = Trainer.Config(
        hf_assets_path=str(tmp_path),
        dump_folder="custom-output",
        training=UpstreamTrainingConfig(
            local_batch_size=3,
            seq_len=4096,
            steps=17,
        ),
    )

    def test_config() -> Trainer.Config:
        return source

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])

    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.extension, ExtensionConfig)
    assert isinstance(config.extension.quantization, QuantizationExtensionConfig)
    assert config.extension.quantization.enable_quantized_training is False
    assert isinstance(config.training, NPUTrainingConfig)
    assert isinstance(config.training.extension, TrainingExtensionConfig)
    assert config.training.extension.allow_hf32 is True
    assert isinstance(config.optimizer, OptimizerConfig)
    assert config.optimizer.name == "native"
    for config_field in fields(Trainer.Config):
        if config_field.name in ("optimizer", "training"):
            continue
        assert getattr(config, config_field.name) == getattr(source, config_field.name)
    for config_field in fields(source.optimizer):
        assert getattr(config.optimizer, config_field.name) == getattr(
            source.optimizer,
            config_field.name,
        )
    for config_field in fields(UpstreamTrainingConfig):
        assert getattr(config.training, config_field.name) == getattr(
            source.training,
            config_field.name,
        )


def test_config_manager_reapplying_patch_does_not_stack_registered_converter(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_reapplied_config_converter_registry"
    source = Trainer.Config(hf_assets_path=str(tmp_path))
    converted_configs: list[Trainer.Config] = []

    def test_config() -> Trainer.Config:
        return source

    def record_conversion(
        _converter: TrainerConfigConverter,
        config: Trainer.Config,
    ) -> Trainer.Config:
        converted_configs.append(config)
        return config

    _install_config_registry(monkeypatch, module_name, test_config)
    monkeypatch.setattr(TrainerConfigConverter, "convert", record_conversion)
    config_manager.apply()
    config_manager.apply()

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])

    assert converted_configs == [source]
    assert isinstance(config, Trainer.Config)


def test_config_manager_parses_training_extension_hf32_option(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_hf32_config_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--training.extension.no-allow-hf32",
        ]
    )

    assert isinstance(config, TrainerEx.Config)
    assert config.training.extension.allow_hf32 is False


def test_config_manager_parses_quantization_extension(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_quantization_config_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

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


def test_config_manager_materializes_muon_from_cli(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_muon_config_registry"
    profile = MuonOptimizerProfile(
        muon_pattern=r"matrix\\.weight",
        optimizer_factory_kwargs={
            "DistributedMuon": {
                "compute_sharding_by_fqn": {},
                "bucket_configs": (),
            }
        },
    )

    def test_config() -> TrainerEx.Config:
        return TrainerEx.Config(
            hf_assets_path=str(tmp_path),
            optimizer=OptimizerConfig(_muon_profile=profile),
        )

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--optimizer.name",
            "Muon",
            "--optimizer.lr",
            "2.2e-4",
            "--optimizer.weight_decay",
            "0.1",
            "--optimizer.muon_momentum",
            "0.95",
            "--optimizer.muon_enable_nesterov",
            "--optimizer.muon_ns_steps",
            "10",
            "--optimizer.muon_adjust_lr_fn",
            "match_rms_adamw",
        ]
    )

    assert config.optimizer.name == "Muon"
    assert len(config.optimizer.param_groups) == 2
    muon_group, adamw_group = config.optimizer.param_groups
    assert muon_group.optimizer_name == "DistributedMuon"
    assert muon_group.optimizer_kwargs["lr"] == pytest.approx(2.2e-4)
    assert muon_group.optimizer_kwargs["momentum"] == pytest.approx(0.95)
    assert muon_group.optimizer_kwargs["nesterov"] is True
    assert muon_group.optimizer_kwargs["ns_steps"] == 10
    assert muon_group.optimizer_kwargs["foreach"] is False
    assert adamw_group.optimizer_name == "AdamW"
    assert adamw_group.optimizer_kwargs["foreach"] is False


def test_config_package_does_not_reexport_trainer_config():
    assert not hasattr(npu_config, "TrainerConfig")


def test_trainer_ex_config_exposes_hf32_only_under_training_extension():
    config = TrainerEx.Config()

    assert not hasattr(config, "allow_hf32")
    assert config.training.extension.allow_hf32 is True


def test_config_manager_preserves_explicit_npu_training_extension(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_explicit_training_extension_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(
            hf_assets_path=str(tmp_path),
            training=NPUTrainingConfig(
                steps=23,
                extension=TrainingExtensionConfig(allow_hf32=False),
            ),
        )

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])

    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.training, NPUTrainingConfig)
    assert config.training.steps == 23
    assert config.training.extension.allow_hf32 is False


def test_config_manager_preserves_specialized_trainer_config(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_specialized_config_registry"

    class SpecializedTrainer(Trainer):
        @dataclass(kw_only=True, slots=True)
        class Config(Trainer.Config):
            specialized_option: int = 7

    built_configs: list[Trainer.Config] = []

    def test_config() -> Trainer.Config:
        return SpecializedTrainer.Config(
            hf_assets_path=str(tmp_path),
            specialized_option=11,
        )

    def capture_config(_self, config: Trainer.Config) -> None:
        built_configs.append(config)

    _install_config_registry(monkeypatch, module_name, test_config)
    monkeypatch.setattr(SpecializedTrainer, "__init__", capture_config)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])
    trainer = config.build()

    assert isinstance(config, SpecializedTrainer.Config)
    assert config.specialized_option == 11
    assert isinstance(trainer, SpecializedTrainer)
    assert len(built_configs) == 1
    assert isinstance(built_configs[0], SpecializedTrainer.Config)


def test_trainer_ex_applies_enabled_quantization_before_base_initialization(monkeypatch):
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

    def apply_hf32(allow_hf32):
        events.append(("apply_hf32", allow_hf32))

    def initialize_base_trainer(_self, base_config):
        events.append(("initialize_base", base_config.model_spec))

    converter_module.apply_quantization_converter = apply_quantization
    monkeypatch.setitem(sys.modules, "interfaces.torchao_converter", converter_module)
    monkeypatch.setattr(trainer_module, "set_allow_hf32", apply_hf32)
    monkeypatch.setattr(Trainer, "__init__", initialize_base_trainer)

    config = TrainerEx.Config(
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

    trainer = config.build()

    assert isinstance(trainer, TrainerEx)
    assert events == [
        ("quantize", source_model_spec, "all_block_fp8", True, 7.0, False),
        ("apply_hf32", True),
        ("initialize_base", quantized_model_spec),
    ]


def test_trainer_ex_skips_disabled_quantization_and_preserves_model_spec(monkeypatch):
    source_model_spec = object()
    initialized_model_specs = []
    converter_module = types.ModuleType("interfaces.torchao_converter")

    def unexpected_quantization(*args, **kwargs):
        raise AssertionError("quantization converter must remain disabled")

    def initialize_base_trainer(_self, base_config):
        initialized_model_specs.append(base_config.model_spec)

    converter_module.apply_quantization_converter = unexpected_quantization
    monkeypatch.setitem(sys.modules, "interfaces.torchao_converter", converter_module)
    monkeypatch.setattr(trainer_module, "set_allow_hf32", lambda _allow_hf32: None)
    monkeypatch.setattr(Trainer, "__init__", initialize_base_trainer)

    config = TrainerEx.Config(model_spec=source_model_spec)
    trainer = config.build()

    assert isinstance(trainer, TrainerEx)
    assert initialized_model_specs == [source_model_spec]


def test_trainer_ex_applies_training_hf32_before_base_initialization(monkeypatch):
    events = []
    config = TrainerEx.Config(
        training=NPUTrainingConfig(
            extension=TrainingExtensionConfig(allow_hf32=False),
        )
    )

    def apply_hf32(allow_hf32):
        events.append(("apply_hf32", allow_hf32))

    def initialize_base_trainer(_self, base_config):
        events.append(("initialize_base", base_config))

    monkeypatch.setattr(trainer_module, "set_allow_hf32", apply_hf32)
    monkeypatch.setattr(Trainer, "__init__", initialize_base_trainer)

    TrainerEx(config)

    assert events == [
        ("apply_hf32", False),
        ("initialize_base", config),
    ]


def test_trainer_ex_config_build_constructs_trainer_ex(monkeypatch):
    built_configs = []

    def capture_config(_self, config):
        built_configs.append(config)

    monkeypatch.setattr(TrainerEx, "__init__", capture_config)

    trainer = TrainerEx.Config(
        training=NPUTrainingConfig(
            extension=TrainingExtensionConfig(allow_hf32=False),
        )
    ).build()

    assert isinstance(trainer, TrainerEx)
    assert len(built_configs) == 1
    assert isinstance(built_configs[0], TrainerEx.Config)
    assert built_configs[0].training.extension.allow_hf32 is False


@pytest.mark.parametrize(
    "allow_hf32",
    [False, True],
    ids=("disabled", "enabled"),
)
def test_set_allow_hf32_updates_all_backends(monkeypatch, allow_hf32):
    fake_torch_npu = types.SimpleNamespace(
        npu=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_hf32=None),
            conv=types.SimpleNamespace(allow_hf32=None),
            aclnn=types.SimpleNamespace(allow_hf32=None),
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    distributed_utils.set_allow_hf32(allow_hf32)

    assert fake_torch_npu.npu.matmul.allow_hf32 is allow_hf32
    assert fake_torch_npu.npu.conv.allow_hf32 is allow_hf32
    assert fake_torch_npu.npu.aclnn.allow_hf32 is allow_hf32
