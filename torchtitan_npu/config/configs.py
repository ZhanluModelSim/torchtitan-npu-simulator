# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed NPU extensions to TorchTitan's training configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Annotated, Any, ClassVar, Literal

import tyro
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import TrainingConfig as _BaseTrainingConfig
from torchtitan.trainer import Trainer

from torchtitan_npu.extensions.trainer import TrainerEx

QuantizationRecipe = Literal["all_mxfp8", "mix", "all_block_fp8"]


@dataclass(frozen=True, slots=True)
class MuonOptimizerProfile:
    """Model-owned metadata required to construct DistributedMuon.

    The profile intentionally excludes scalar optimizer hyperparameters. Those
    are public CLI fields on :class:`OptimizerConfig` and are materialized only
    after Tyro has applied command-line overrides.
    """

    muon_pattern: str
    optimizer_factory_kwargs: Mapping[str, Mapping[str, Any]]


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(OptimizersContainer.Config):
    """NPU optimizer CLI schema while preserving native optimizer configs."""

    name: Literal["native", "Muon"] = "native"
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_enable_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw", "spectral_unclamped"] = "match_rms_adamw"
    muon_ns_coefficients: tuple[float, float, float] = (
        3.4445,
        -4.7750,
        2.0315,
    )
    muon_eps: float = 1e-7
    _muon_profile: Annotated[MuonOptimizerProfile | None, tyro.conf.Suppress] = None

    def materialize(self) -> None:
        """Turn an explicit Muon selection into upstream optimizer groups.

        ``native`` is intentionally a strict no-op so converting every NPU
        recipe to this schema cannot alter its existing optimizer behavior.
        """
        if self.name == "native":
            return
        if self._muon_profile is None:
            raise ValueError("optimizer.name=Muon requires a recipe with a DSV4 Muon profile")

        self.param_groups = [
            ParamGroupConfig(
                pattern=self._muon_profile.muon_pattern,
                optimizer_name="DistributedMuon",
                optimizer_kwargs={
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "momentum": self.muon_momentum,
                    "nesterov": self.muon_enable_nesterov,
                    "ns_steps": self.muon_ns_steps,
                    "adjust_lr_fn": self.muon_adjust_lr_fn,
                    "ns_coefficients": self.muon_ns_coefficients,
                    "eps": self.muon_eps,
                    "fused": False,
                    "foreach": False,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": self.lr,
                    "betas": (self.beta1, self.beta2),
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                    "fused": False,
                    "foreach": False,
                },
            ),
        ]
        self.optimizer_factory_kwargs_by_name = {
            name: dict(kwargs) for name, kwargs in self._muon_profile.optimizer_factory_kwargs.items()
        }


@dataclass(kw_only=True, slots=True)
class QuantizationExtensionConfig:
    """TorchAO-NPU quantized-training options.

    These fields define the public CLI schema. The quantization integration can
    consume them after CLI parsing without adding model-specific options to the
    upstream TorchTitan configuration.
    """

    enable_quantized_training: bool = False
    recipe: QuantizationRecipe = "mix"
    enable_mxfp4_qat: bool = False
    dst_type_max: float = 0.0


@dataclass(kw_only=True, slots=True)
class ExtensionConfig:
    """Global NPU extensions without an upstream component owner.

    Add a semantic group as a nested dataclass, then expose it with
    ``field(default_factory=...)``. For example::

        @dataclass(kw_only=True, slots=True)
        class RuntimeExtensionConfig:
            enable_feature: bool = False

        @dataclass(kw_only=True, slots=True)
        class ExtensionConfig:
            runtime: RuntimeExtensionConfig = field(
                default_factory=RuntimeExtensionConfig,
            )

    This produces the CLI option ``--extension.runtime.enable-feature``.
    """

    quantization: QuantizationExtensionConfig = field(
        default_factory=QuantizationExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class TrainingExtensionConfig:
    """
    NPU extensions owned by the training configuration.
    """

    allow_hf32: bool = True
    """
    Enable HF32 for the NPU matmul, convolution, and ACLNN backends.
    """


@dataclass(kw_only=True, slots=True)
class TrainingConfig(_BaseTrainingConfig):
    """Training options that are specific to NPU execution."""

    extension: TrainingExtensionConfig = field(
        default_factory=TrainingExtensionConfig,
    )


def _dataclass_values(source: object, target_type: type[Any]) -> dict[str, Any]:
    """Collect init fields shared by two dataclass configuration types."""

    if not is_dataclass(source) or isinstance(source, type):
        raise TypeError(f"{type(source).__name__} must be a dataclass instance")
    if not is_dataclass(target_type):
        raise TypeError(f"{target_type.__name__} must be a dataclass type")

    target_fields = {config_field.name for config_field in fields(target_type) if config_field.init}
    return {
        config_field.name: getattr(source, config_field.name)
        for config_field in fields(source)
        if config_field.init and config_field.name in target_fields
    }


def _convert_config(source: object, target_type: type[Any]) -> Any:
    """Convert an upstream config to an extension config when necessary."""

    if isinstance(source, target_type):
        return source
    return target_type(**_dataclass_values(source, target_type))


@dataclass(kw_only=True, slots=True)
class TrainerConfig(TrainerEx.Config):
    """The standard TorchTitan trainer config with NPU training settings."""

    _CONFIG_EXTENSIONS: ClassVar[dict[str, type[Any]]] = {
        "optimizer": OptimizerConfig,
        "training": TrainingConfig,
    }

    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    optimizer: OptimizerConfig = field(  # pyrefly: ignore [bad-override]
        default_factory=OptimizerConfig
    )
    training: TrainingConfig = field(  # pyrefly: ignore [bad-override]
        default_factory=TrainingConfig
    )

    def __post_init__(self) -> None:
        # ``slots=True`` dataclasses are recreated by the decorator, so a
        # zero-argument ``super()`` can retain the pre-decoration class cell.
        # TrainerEx.Config currently adds no post-init behavior.
        Trainer.Config.__post_init__(self)
        self.optimizer.materialize()
        if self.optimizer.name == "Muon" and (
            self.parallelism.tensor_parallel_degree > 1 or self.parallelism.pipeline_parallel_degree > 1
        ):
            raise ValueError(
                "DeepSeek-V4 DistributedMuon requires "
                "tensor_parallel_degree=1 and pipeline_parallel_degree=1; "
                "TP _StridedShard and PP stage-local parameter groups are not admitted yet"
            )

    @classmethod
    def from_trainer_config(cls, config: Trainer.Config) -> "TrainerConfig":
        """Wrap an upstream trainer config with the NPU config schema."""

        if isinstance(config, cls):
            return config

        values = _dataclass_values(config, cls)
        for field_name, target_type in cls._CONFIG_EXTENSIONS.items():
            values[field_name] = _convert_config(
                getattr(config, field_name),
                target_type,
            )
        return cls(**values)

    def build(self, **kwargs):
        """Apply NPU training settings before constructing the trainer."""

        from torchtitan_npu.distributed.utils import set_allow_hf32

        quantization_config = self.extension.quantization
        if quantization_config.enable_quantized_training:
            from interfaces.torchao_converter import apply_quantization_converter

            model_compile_enabled = self.compile.enable and "model" in self.compile.components
            self.model_spec = apply_quantization_converter(
                self.model_spec,
                quantization_config,
                model_compile_enabled=model_compile_enabled,
            )
        set_allow_hf32(self.training.extension.allow_hf32)
        return TrainerEx.Config.build(self, **kwargs)
