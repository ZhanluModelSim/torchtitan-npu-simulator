# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, fields

import pytest
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.config import ConfigManager
from torchtitan.config import TrainingConfig as UpstreamTrainingConfig
from torchtitan.models.llama3 import model_registry
from torchtitan.trainer import Trainer

import torchtitan_npu.config as npu_config
import torchtitan_npu.extensions.torchtitan.trainer as trainer_module
from torchtitan_npu.config import (
    ExtensionConfig,
    MuonOptimizerProfile,
    OptimConfig,
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


def _debug_model():
    return model_registry("debugmodel")


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
        model=_debug_model(),
        hf_assets_path=str(tmp_path),
        dump_folder="custom-output",
        training=UpstreamTrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=3 * 4096,
            num_tokens_per_train_step=6 * 4096,
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
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    assert config.optim.optimizer.name == "AdamW"
    for config_field in fields(Trainer.Config):
        if config_field.name in ("optim", "profiler", "training", "checkpointer"):
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
    assert config.checkpointer is None
    for config_field in fields(source.optim.optimizer):
        assert getattr(config.optim.optimizer, config_field.name) == getattr(
            source.optim.optimizer,
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
    source = Trainer.Config(model=_debug_model(), hf_assets_path=str(tmp_path))
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
        return Trainer.Config(model=_debug_model(), hf_assets_path=str(tmp_path))

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
        return Trainer.Config(model=_debug_model(), hf_assets_path=str(tmp_path))

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
        return Trainer.Config(model=_debug_model(), hf_assets_path=str(tmp_path))

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


def test_config_manager_splits_comma_joined_fsdp_prequantize_fqns(monkeypatch, tmp_path):
    """The A5 launch scripts pass the whitelist as one comma-joined argument."""
    module_name = "_torchtitan_npu_prequantize_fqns_config_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(model=_debug_model(), hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--extension.quantization.enable-fsdp-prequantize",
            "--extension.quantization.fsdp-prequantize-fqns",
            ".attention.wq_b,.attention.wo_a",
        ]
    )

    assert config.extension.quantization.fsdp_prequantize_fqns == [".attention.wq_b", ".attention.wo_a"]


def test_config_manager_rejects_space_separated_fsdp_prequantize_fqns(monkeypatch, tmp_path):
    """The old space-separated multi-value form leaves a stray token and must not parse."""
    module_name = "_torchtitan_npu_prequantize_fqns_space_config_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(model=_debug_model(), hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

    with pytest.raises(SystemExit) as exit_info:
        ConfigManager().parse_args(
            [
                "--module",
                module_name,
                "--config",
                "test_config",
                "--extension.quantization.enable-fsdp-prequantize",
                "--extension.quantization.fsdp-prequantize-fqns",
                ".attention.wq_b",
                ".attention.wo_a",
            ]
        )

    assert exit_info.value.code != 0


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
            model=_debug_model(),
            hf_assets_path=str(tmp_path),
            optim=OptimConfig(optimizer=OptimizerConfig(_muon_profile=profile)),
        )

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        [
            "--module",
            module_name,
            "--config",
            "test_config",
            "--optim.optimizer.name",
            "Muon",
            "--optim.optimizer.lr",
            "2.2e-4",
            "--optim.optimizer.weight_decay",
            "0.1",
            "--optim.optimizer.muon_momentum",
            "0.95",
            "--optim.optimizer.muon_enable_nesterov",
            "--optim.optimizer.muon_ns_steps",
            "10",
            "--optim.optimizer.muon_adjust_lr_fn",
            "match_rms_adamw",
        ]
    )

    assert config.optim.optimizer.name == "Muon"
    assert len(config.optim.optimizer.optimizers) == 2
    muon_group, adamw_group = config.optim.optimizer.optimizers
    assert type(muon_group).__qualname__ == "DistMuon.Config"
    assert muon_group.lr == pytest.approx(2.2e-4)
    assert muon_group.momentum == pytest.approx(0.95)
    assert muon_group.nesterov is True
    assert muon_group.ns_steps == 10
    assert isinstance(adamw_group, AdamW.Config)
    assert adamw_group.foreach is False


def test_config_manager_accepts_adamw_from_cli(monkeypatch, tmp_path):
    module_name = "_torchtitan_npu_adamw_config_registry"

    def test_config() -> TrainerEx.Config:
        return TrainerEx.Config(model=_debug_model(), hf_assets_path=str(tmp_path))

    _install_config_registry(monkeypatch, module_name, test_config)

    config = ConfigManager().parse_args(
        ["--module", module_name, "--config", "test_config", "--optim.optimizer.name", "AdamW"]
    )

    assert config.optim.optimizer.name == "AdamW"


def test_config_package_does_not_reexport_trainer_config():
    assert not hasattr(npu_config, "TrainerConfig")


def test_trainer_ex_config_exposes_hf32_only_under_training_extension():
    config = TrainerEx.Config(
        model=_debug_model(),
    )

    assert not hasattr(config, "allow_hf32")
    assert config.training.extension.allow_hf32 is True


def test_config_manager_preserves_explicit_npu_training_extension(
    monkeypatch,
    tmp_path,
):
    module_name = "_torchtitan_npu_explicit_training_extension_registry"

    def test_config() -> Trainer.Config:
        return Trainer.Config(
            model=_debug_model(),
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
            model=_debug_model(),
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
            model=_debug_model(),
            hf_assets_path=str(tmp_path),
            compile=GraphTrainerCompileConfig(),
            training=UpstreamTrainingConfig(steps=23),
        )

    _install_config_registry(monkeypatch, module_name, test_config)

    def initialize_graph_trainer(trainer, config):
        trainer.engine = types.SimpleNamespace(model_parts=[])
        trainer.gradient_accumulation_steps = 1

    monkeypatch.setattr(trainer_module, "set_allow_hf32", applied_hf32.append)
    monkeypatch.setattr(GraphTrainer, "__init__", initialize_graph_trainer)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])
    trainer = config.build()

    assert isinstance(config, GraphTrainerEx.Config)
    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert isinstance(config.training, NPUTrainingConfig)
    assert config.training.steps == 23
    assert config.training.extension.allow_hf32 is True
    assert isinstance(trainer, GraphTrainerEx)
    assert applied_hf32 == [True]


def test_trainer_ex_applies_enabled_quantization_before_base_initialization(monkeypatch):
    events = []
    source_model = _debug_model()
    quantized_model = _debug_model()
    converter_module = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.converter")

    def apply_quantization(model, quantization_config, *, model_compile_enabled, override_config=None):
        events.append(
            (
                "quantize",
                model,
                quantization_config.recipe,
                quantization_config.enable_mxfp4_qat,
                quantization_config.dst_type_max,
                quantization_config.enable_fsdp_prequantize,
                model_compile_enabled,
            )
        )
        return quantized_model

    def apply_hf32(allow_hf32):
        events.append(("apply_hf32", allow_hf32))

    def initialize_base_trainer(trainer, base_config):
        trainer.engine = types.SimpleNamespace(model_parts=[])
        trainer.gradient_accumulation_steps = 1
        events.append(("initialize_base", base_config.model))

    converter_module.apply_quantization_converter = apply_quantization
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.adaptor.torchao.converter", converter_module)
    monkeypatch.setattr(trainer_module, "set_allow_hf32", apply_hf32)
    monkeypatch.setattr(TrainerEx.__base__, "__init__", initialize_base_trainer)

    config = TrainerEx.Config(
        model=source_model,
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
        ("quantize", source_model, "all_block_fp8", True, 7.0, False, False),
        ("apply_hf32", True),
        ("initialize_base", quantized_model),
    ]


def test_trainer_ex_skips_disabled_quantization_and_preserves_model(monkeypatch):
    source_model = _debug_model()
    initialized_models = []
    converter_module = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.converter")

    def unexpected_quantization(*args, **kwargs):
        raise AssertionError("quantization converter must remain disabled")

    def initialize_base_trainer(trainer, base_config):
        trainer.engine = types.SimpleNamespace(model_parts=[])
        trainer.gradient_accumulation_steps = 1
        initialized_models.append(base_config.model)

    converter_module.apply_quantization_converter = unexpected_quantization
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.adaptor.torchao.converter", converter_module)
    monkeypatch.setattr(trainer_module, "set_allow_hf32", lambda _allow_hf32: None)
    monkeypatch.setattr(TrainerEx.__base__, "__init__", initialize_base_trainer)

    config = TrainerEx.Config(model=source_model)
    trainer = config.build()

    assert isinstance(trainer, TrainerEx)
    assert initialized_models == [source_model]


def test_trainer_ex_applies_training_hf32_before_base_initialization(monkeypatch):
    events = []
    config = TrainerEx.Config(
        model=_debug_model(),
        training=NPUTrainingConfig(
            extension=TrainingExtensionConfig(allow_hf32=False),
        ),
    )

    def apply_hf32(allow_hf32):
        events.append(("apply_hf32", allow_hf32))

    def initialize_base_trainer(trainer, base_config):
        trainer.engine = types.SimpleNamespace(model_parts=[])
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
        model=_debug_model(),
        training=NPUTrainingConfig(
            extension=TrainingExtensionConfig(allow_hf32=False),
        ),
    ).build()

    assert isinstance(trainer, TrainerEx)
    assert len(built_configs) == 1
    assert isinstance(built_configs[0], TrainerEx.Config)
    assert built_configs[0].training.extension.allow_hf32 is False


def test_config_manager_preserves_nested_optimization_policy(monkeypatch, tmp_path):
    from torchtitan.components.optim import EMA, LRSchedulersContainer

    module_name = "_torchtitan_npu_nested_optim_registry"
    adamw = AdamW.Config(pattern=r".*", lr=0.003, betas=(0.8, 0.99), fused=False)
    source = Trainer.Config(
        model=_debug_model(),
        hf_assets_path=str(tmp_path),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(optimizers=[adamw]),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=17, total_steps=91),
            ema=EMA.Config(),
            max_norm=0.25,
            enable_cuda_graph=True,
        ),
        checkpointer=CheckpointManager.Config(interval=13),
        create_seed_checkpoint=True,
    )
    _install_config_registry(monkeypatch, module_name, lambda: source)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])

    assert isinstance(config.optim, OptimConfig)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    assert config.optim.optimizer.optimizers == [adamw]
    assert config.optim.lr_scheduler.warmup_steps == 17
    assert config.optim.lr_scheduler.total_steps == 91
    from torchtitan_npu.extensions.torchtitan.components.optimizer.ema import EMA as NPUEMA

    assert type(config.optim.ema) is NPUEMA.Config
    assert type(source.optim.ema) is EMA.Config
    assert all(
        getattr(config.optim.ema, field.name) == getattr(source.optim.ema, field.name) for field in fields(EMA.Config)
    )
    assert config.optim.max_norm == 0.25
    assert config.optim.enable_cuda_graph is True
    assert config.checkpointer.interval == 13
    assert config.create_seed_checkpoint is True
    assert config.compile is None


def test_config_manager_preserves_recipe_optimizer_type_and_fields(monkeypatch, tmp_path):
    class RecipeOptimizer(OptimizersContainer):
        @dataclass(kw_only=True, slots=True)
        class Config(OptimizersContainer.Config):
            recipe_policy: str = "specialized"

    module_name = "_torchtitan_npu_recipe_optimizer_registry"
    source_optimizer = RecipeOptimizer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=0.007)],
        recipe_policy="retain-me",
    )
    source = Trainer.Config(
        model=_debug_model(),
        hf_assets_path=str(tmp_path),
        optim=Optim.Config(optimizer=source_optimizer),
    )
    _install_config_registry(monkeypatch, module_name, lambda: source)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "test_config"])

    assert type(config.optim.optimizer) is RecipeOptimizer.Config
    assert config.optim.optimizer.recipe_policy == "retain-me"
    assert config.optim.optimizer.optimizers[0].lr == 0.007
    assert config.optim.optimizer._owner is RecipeOptimizer


def test_config_manager_overrides_recipe_owned_optional_components(monkeypatch, tmp_path):
    from torchtitan_npu.config import CompileConfig, ModelCompileConfig
    from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager as NPUCheckpointManager

    @dataclass(kw_only=True, slots=True)
    class RecipeCompileConfig(CompileConfig):
        recipe_policy: str = "model-only"

    class RecipeCheckpointer(NPUCheckpointManager):
        @dataclass(kw_only=True, slots=True)
        class Config(NPUCheckpointManager.Config):
            recipe_policy: str = "retain-me"

    module_name = "_torchtitan_npu_optional_owner_registry"
    source = TrainerEx.Config(
        model=_debug_model(),
        hf_assets_path=str(tmp_path),
        compile=RecipeCompileConfig(components=[], model=ModelCompileConfig()),
        checkpointer=RecipeCheckpointer.Config(interval=11, folder="recipe-checkpoints"),
    )
    _install_config_registry(monkeypatch, module_name, lambda: source)
    base = ["--module", module_name, "--config", "test_config"]

    configured = ConfigManager().parse_args(
        [
            *base,
            "--compile.model.backend",
            "aot_eager",
            "--compile.model.no-fullgraph",
            "--checkpointer.interval",
            "17",
            "--checkpointer.extensions.verify-hash-manifest",
        ]
    )

    assert type(configured.compile) is RecipeCompileConfig
    assert configured.compile.components == []
    assert configured.compile.recipe_policy == "model-only"
    assert isinstance(configured.compile.model, ModelCompileConfig)
    assert configured.compile.model.backend == "aot_eager"
    assert configured.compile.model.fullgraph is False
    assert type(configured.checkpointer) is RecipeCheckpointer.Config
    assert configured.checkpointer._owner is RecipeCheckpointer
    assert configured.checkpointer.recipe_policy == "retain-me"
    assert configured.checkpointer.folder == "recipe-checkpoints"
    assert configured.checkpointer.interval == 17
    assert configured.checkpointer.extensions.verify_hash_manifest is True
    assert source.compile.model.backend == "inductor"
    assert source.compile.model.fullgraph is True
    assert source.checkpointer.interval == 11
    assert source.checkpointer.extensions.verify_hash_manifest is False
