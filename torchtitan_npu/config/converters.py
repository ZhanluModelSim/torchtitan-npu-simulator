# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert upstream trainer configs to NPU extension schemas."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from torchtitan.config import derive
from torchtitan.trainer import Trainer


class TrainerConfigConverter:
    """Convert a trainer config and its components to a target schema."""

    def __init__(
        self,
        *,
        target_type: type[Trainer.Config],
        component_types: Mapping[str, type[Any]],
    ) -> None:
        self._target_type = target_type
        self._component_types = component_types

    @staticmethod
    def _convert_component_config(source: object, target_type: type[Any]) -> Any:
        """Convert a config to a registered extension type when necessary."""
        if not isinstance(target_type, type) or not is_dataclass(target_type):
            raise TypeError(f"{target_type!r} must be a dataclass type")
        if isinstance(source, target_type):
            return source

        source_type: type[Any] = type(source)
        if not is_dataclass(source) or isinstance(source, type):
            raise TypeError(f"{source_type.__name__} must be a dataclass instance")

        target_fields = {config_field.name for config_field in fields(target_type) if config_field.init}
        values = {
            config_field.name: getattr(source, config_field.name)
            for config_field in fields(source)
            if config_field.init and config_field.name in target_fields
        }
        return target_type(**values)

    def convert(self, config: Trainer.Config) -> Trainer.Config:
        """Adapt a standard trainer config to the configured target schema."""
        if isinstance(config, self._target_type):
            return config

        deltas = {
            field_name: self._convert_component_config(
                getattr(config, field_name),
                component_type,
            )
            for field_name, component_type in self._component_types.items()
        }
        return derive(config, self._target_type, **deltas)
