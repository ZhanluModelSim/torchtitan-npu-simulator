# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register trainer config converters and install them before Tyro parsing."""

from functools import wraps
from typing import cast

from torchtitan.config.manager import ConfigManager
from torchtitan.trainer import Trainer

from torchtitan_npu.config.converters import TrainerConfigConverter

_original_load_config = ConfigManager._load_config

_CONFIG_CONVERTERS: dict[type[object], TrainerConfigConverter] = {}


def register_config_converter(
    config_type: type[Trainer.Config],
    converter: TrainerConfigConverter,
) -> None:
    """Register a converter for a Trainer config type and its config bases."""
    _CONFIG_CONVERTERS[config_type] = converter
    # Re-exported upstream recipes can still return the base Trainer.Config.
    # Register config bases too, while retaining an explicitly registered
    # converter for any more specialized recipe-owned type.
    for base in config_type.__mro__[1:]:
        if base is object:
            break
        _CONFIG_CONVERTERS.setdefault(base, converter)


@wraps(_original_load_config)
def _patched_load_config(self, args: list[str]) -> tuple[object, list[str]]:
    config, filtered_args = _original_load_config(self, args)
    # A recipe-owned Trainer subclass still needs the registered component
    # adapters. Prefer its nearest registered base and preserve its owner.
    converter = next((_CONFIG_CONVERTERS[base] for base in type(config).__mro__ if base in _CONFIG_CONVERTERS), None)
    if converter is not None:
        config = converter.convert(cast("Trainer.Config", config))
    return config, filtered_args


def apply() -> None:
    """Install the loader wrapper that dispatches registered config converters."""
    ConfigManager._load_config = _patched_load_config  # type: ignore[method-assign]


apply()
