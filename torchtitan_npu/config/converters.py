# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert upstream trainer configs to NPU extension schemas."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

from torchtitan.config import Configurable
from torchtitan.trainer import Trainer


class TrainerConfigConverter:
    """Adapt registered component paths without erasing recipe-owned configs."""

    def __init__(
        self,
        *,
        target_type: type[Trainer.Config],
        component_types: Mapping[str, type[Any]],
    ) -> None:
        self._target_type = target_type
        self._component_types = dict(component_types)

    @staticmethod
    def _convert_component_config(
        source: object,
        target_type: type[Any],
        **updates: Any,
    ) -> Any:
        """Promote a base config, preserving specialized owners and all fields."""
        if not isinstance(target_type, type) or not is_dataclass(target_type):
            raise TypeError(f"{target_type!r} must be a dataclass type")
        if source is None:
            return None
        source_type = type(source)
        if not is_dataclass(source) or isinstance(source, type):
            raise TypeError(f"{source_type.__name__} must be a dataclass instance")
        if isinstance(source, target_type):
            return replace(source, **updates) if updates else source
        # Foreign component schemas own their build() implementation. Replacing
        # one with a sibling NPU schema could discard both its fields and owner.
        if isinstance(source, Configurable.Config) and not issubclass(target_type, source_type):
            return replace(source, **updates) if updates else source
        target_fields = {config_field.name for config_field in fields(target_type) if config_field.init}
        source_fields = {config_field.name for config_field in fields(source) if config_field.init}
        dropped = source_fields - target_fields
        if dropped:
            raise ValueError(
                f"Converting {source_type.__qualname__} to {target_type.__qualname__} "
                f"would discard fields: {sorted(dropped)}"
            )
        values = {name: getattr(source, name) for name in source_fields}
        values.update(updates)
        return target_type(**values)

    def convert(self, config: object, path: str = "") -> Any:
        """Adapt a trainer tree, including optional and nested components."""
        if config is None:
            return None
        prefix = f"{path}." if path else ""
        child_names = {
            registered_path.removeprefix(prefix).split(".", 1)[0]
            for registered_path in self._component_types
            if registered_path.startswith(prefix) and registered_path != path
        }
        updates = {}
        for name in child_names:
            if not hasattr(config, name):
                raise ValueError(f"Config component path {prefix + name!r} does not exist")
            value = getattr(config, name)
            converted = self.convert(value, prefix + name)
            if converted is not value:
                updates[name] = converted
        target_type = self._target_type if not path else self._component_types.get(path)
        if target_type is not None:
            return self._convert_component_config(config, target_type, **updates)
        return replace(config, **updates) if updates else config
