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
import torchtitan_npu.extensions.torchtitan.trainer as trainer_module
from torchtitan_npu.config import (
    ExtensionConfig,
    MuonOptimizerProfile,
    OptimizerConfig,
    QuantizationExtensionConfig,
    TrainingExtensionConfig,
)
from torchtitan_npu.config import TrainingConfig as NPUTrainingConfig
from torchtitan_npu.config import manager as config_manager
from torchtitan_npu.config.configs import KVNormQuantizationConfig
from torchtitan_npu.config.converters import TrainerConfigConverter
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import GraphTrainerEx
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


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
    assert config.optimizer.name == "AdamW"
    for config_field in fields(Trainer.Config):
        if config_field.name in ("optimizer", "profiler", "training", "checkpoint"):
            continue
        assert getattr(config, config_field.name) == getattr(source, config_field.name)
    assert isinstance(config.profiler, CANNProfiler.Config)
    assert isinstance(config.profiler.build(), CANNProfiler)
    for config_field in fields(source.profiler):
        assert getattr(config.profiler, config_field.name) == getattr(
            source.profiler,
            config_field.name,
        )
    assert config.profiler.extension.profiler_start is None
    assert config.profiler.extension.profiler_end is None
    assert config.profiler.extension.profile_ranks == [-1]
    assert config.profiler.extension.profile_with_memory is False
    assert config.profiler.extension.profile_with_stack is False
    assert config.profiler.extension.enable_online_parse is True
    for config_field in fields(source.checkpoint):
        assert getattr(config.checkpoint, config_field.name) == getattr(
            source.checkpoint,
            config_field.name,
        )
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


def test_config_manager_parses_cann_profiler_without_override(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_profiler_config_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--profiler.enable-profiling",
            "--profiler.extension.profiler-start",
            "5",
            "--profiler.extension.profiler-end",
            "8",
            "--profiler.profiler-warmup",
            "3",
            "--profiler.extension.profile-ranks",
            "0",
            "--profiler.extension.profile-with-memory",
            "--profiler.extension.profile-with-stack",
            "--profiler.extension.no-enable-online-parse",
        ]
    )

    assert isinstance(config.profiler, CANNProfiler.Config)
    assert config.override.imports == []
    assert config.profiler.enable_profiling is True
    assert config.profiler.extension.profiler_start == 5
    assert config.profiler.extension.profiler_end == 8
    assert config.profiler.profiler_warmup == 3
    assert config.profiler.extension.profile_ranks == [0]
    assert config.profiler.extension.profile_with_memory is True
    assert config.profiler.extension.profile_with_stack is True
    assert config.profiler.extension.enable_online_parse is False


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
            "--extension.quantization.kv-norm-quantization.format",
            "mxfp8",
            "--extension.quantization.kv-norm-quantization.fqns",
            ".attention.kv_norm,.attention.compressor.norm",
            "--extension.quantization.kv-norm-quantization.block-size",
            "64",
        ]
    )

    quantization = config.extension.quantization
    assert quantization.enable_quantized_training is True
    assert quantization.recipe == "all_block_fp8"
    assert quantization.enable_mxfp4_qat is True
    assert quantization.dst_type_max == 7.0
    assert quantization.enable_fsdp_prequantize is False
    assert quantization.kv_norm_quantization.format == "mxfp8"
    assert quantization.kv_norm_quantization.fqns == [".attention.kv_norm", ".attention.compressor.norm"]
    assert quantization.kv_norm_quantization.block_size == 64


def test_kv_norm_quantization_config_validates_its_format():
    """
    Only the format is checked here; the block size is validated by the MX
    configs that consume it (``MXQuantizeConfig.__post_init__``).
    """
    assert KVNormQuantizationConfig().format is None
    assert KVNormQuantizationConfig(format="mxfp8", block_size=64).block_size == 64

    with pytest.raises(ValueError, match="format must be None or one of: mxfp8"):
        KVNormQuantizationConfig(format="fp8")  # pyrefly: ignore [bad-argument-type]


def test_config_manager_materializes_muon_from_cli(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_muon_config_registry"
    profile = MuonOptimizerProfile(
        muon_pattern=r"matrix\\.weight",
        optimizer_factory_kwargs={
            "DistMuon": {
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
    assert muon_group.optimizer_name == "DistMuon"
    assert muon_group.optimizer_kwargs["lr"] == pytest.approx(2.2e-4)
    assert muon_group.optimizer_kwargs["momentum"] == pytest.approx(0.95)
    assert muon_group.optimizer_kwargs["nesterov"] is True
    assert muon_group.optimizer_kwargs["ns_steps"] == 10
    assert muon_group.optimizer_kwargs["foreach"] is False
    assert adamw_group.optimizer_name == "AdamW"
    assert adamw_group.optimizer_kwargs["foreach"] is False


def test_config_manager_accepts_adamw_from_cli(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_adamw_config_registry"

    def test_config() -> TrainerEx.Config:
        return TrainerEx.Config(hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        ["--module", module_name, "--config", "test_config", "--optimizer.name", "AdamW"]
    )

    assert config.optimizer.name == "AdamW"


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


def test_config_manager_builds_graph_trainer_ex_and_applies_hf32(monkeypatch, tmp_path):
    # Import after torchtitan_npu applies its Trainer patches.
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    module_name = "_torchtitan_npu_graph_trainer_config_registry"
    applied_hf32 = []

    def test_config() -> GraphTrainer.Config:
        return GraphTrainer.Config(
            hf_assets_path=str(tmp_path),
            compile=GraphTrainerCompileConfig(enable=True),
            training=UpstreamTrainingConfig(steps=23),
        )

    _install_config_registry(monkeypatch, module_name, test_config)

    def initialize_graph_trainer(trainer, config):
        trainer.model_parts = []
        trainer.gradient_accumulation_steps = 1

    monkeypatch.setattr(trainer_module, "set_allow_hf32", applied_hf32.append)
    monkeypatch.setattr(GraphTrainer, "__init__", initialize_graph_trainer)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])
    trainer = config.build()

    assert isinstance(config, GraphTrainerEx.Config)
    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert config.compile.enable is True
    assert isinstance(config.training, NPUTrainingConfig)
    assert config.training.steps == 23
    assert config.training.extension.allow_hf32 is True
    assert isinstance(trainer, GraphTrainerEx)
    assert applied_hf32 == [True]


def test_trainer_ex_applies_enabled_quantization_before_base_initialization(monkeypatch):
    events = []
    source_model_spec = object()
    quantized_model_spec = object()
    converter_module = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.converter")

    def apply_quantization(model_spec, quantization_config, *, model_compile_enabled):
        events.append(
            (
                "quantize",
                model_spec,
                quantization_config.recipe,
                quantization_config.enable_mxfp4_qat,
                quantization_config.dst_type_max,
                quantization_config.enable_fsdp_prequantize,
                model_compile_enabled,
            )
        )
        return quantized_model_spec

    def apply_hf32(allow_hf32):
        events.append(("apply_hf32", allow_hf32))

    def initialize_base_trainer(trainer, base_config):
        trainer.model_parts = []
        trainer.gradient_accumulation_steps = 1
        events.append(("initialize_base", base_config.model_spec))

    converter_module.apply_quantization_converter = apply_quantization
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.adaptor.torchao.converter", converter_module)
    monkeypatch.setattr(trainer_module, "set_allow_hf32", apply_hf32)
    monkeypatch.setattr(TrainerEx.__base__, "__init__", initialize_base_trainer)

    config = TrainerEx.Config(
        model_spec=source_model_spec,
        extension=ExtensionConfig(
            quantization=QuantizationExtensionConfig(
                enable_quantized_training=True,
                recipe="all_block_fp8",
                enable_mxfp4_qat=True,
                dst_type_max=7.0,
                enable_fsdp_prequantize=False,
            ),
        ),
    )

    trainer = config.build()

    assert isinstance(trainer, TrainerEx)
    assert events == [
        ("quantize", source_model_spec, "all_block_fp8", True, 7.0, False, False),
        ("apply_hf32", True),
        ("initialize_base", quantized_model_spec),
    ]


def test_trainer_ex_skips_disabled_quantization_and_preserves_model_spec(monkeypatch):
    source_model_spec = object()
    initialized_model_specs = []
    converter_module = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.converter")

    def unexpected_quantization(*args, **kwargs):
        raise AssertionError("quantization converter must remain disabled")

    def initialize_base_trainer(trainer, base_config):
        trainer.model_parts = []
        trainer.gradient_accumulation_steps = 1
        initialized_model_specs.append(base_config.model_spec)

    converter_module.apply_quantization_converter = unexpected_quantization
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.adaptor.torchao.converter", converter_module)
    monkeypatch.setattr(trainer_module, "set_allow_hf32", lambda _allow_hf32: None)
    monkeypatch.setattr(TrainerEx.__base__, "__init__", initialize_base_trainer)

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

    def initialize_base_trainer(trainer, base_config):
        trainer.model_parts = []
        trainer.gradient_accumulation_steps = 1
        events.append(("initialize_base", base_config))

    monkeypatch.setattr(trainer_module, "set_allow_hf32", apply_hf32)
    monkeypatch.setattr(TrainerEx.__base__, "__init__", initialize_base_trainer)

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
