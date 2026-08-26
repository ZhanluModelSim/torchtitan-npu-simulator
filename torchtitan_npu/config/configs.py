# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed NPU extensions to TorchTitan's training configuration."""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar

from torchtitan.config import TrainingConfig as _BaseTrainingConfig
from torchtitan.trainer import Trainer


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
class TrainerConfig(Trainer.Config):
    """The standard TorchTitan trainer config with NPU training settings."""

    _CONFIG_EXTENSIONS: ClassVar[dict[str, type[Any]]] = {
        "training": TrainingConfig,
    }

    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    training: TrainingConfig = field(  # pyrefly: ignore [bad-override]
        default_factory=TrainingConfig
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

        set_allow_hf32(self.training.extension.allow_hf32)
        return Trainer.Config.build(self, **kwargs)
