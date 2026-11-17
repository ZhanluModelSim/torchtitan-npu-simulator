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
    # ``patches.torchtitan.trainer`` can replace ``torchtitan.trainer.Trainer``
    # during package activation.  A config registry imported before that
    # replacement may still return the original base ``Trainer.Config``.
    # Register its config bases as well, while retaining an explicitly
    # registered converter for a more specialized type.
    for base in config_type.__mro__[1:]:
        if base is object:
            break
        _CONFIG_CONVERTERS.setdefault(base, converter)


@wraps(_original_load_config)
def _patched_load_config(self, args: list[str]) -> tuple[object, list[str]]:
    config, filtered_args = _original_load_config(self, args)
    converter = _CONFIG_CONVERTERS.get(type(config))
    if converter is not None:
        config = converter.convert(cast("Trainer.Config", config))
    return config, filtered_args


def apply() -> None:
    """Install the loader wrapper that dispatches registered config converters."""
    ConfigManager._load_config = _patched_load_config  # type: ignore[method-assign]


apply()
