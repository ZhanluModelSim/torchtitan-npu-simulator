# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register trainer config converters and install them before Tyro parsing."""

from functools import wraps

from torchtitan.config.manager import ConfigManager
from torchtitan.trainer import Trainer

from torchtitan_npu.config.converters import TrainerConfigConverter

_original_load_config = ConfigManager._load_config

_CONFIG_CONVERTERS: dict[type[Trainer.Config], TrainerConfigConverter] = {}


def register_config_converter(
    config_type: type[Trainer.Config],
    converter: TrainerConfigConverter,
) -> None:
    """Register a converter for one exact Trainer config type."""
    _CONFIG_CONVERTERS[config_type] = converter


@wraps(_original_load_config)
def _patched_load_config(self, args: list[str]) -> tuple[object, list[str]]:
    config, filtered_args = _original_load_config(self, args)
    if isinstance(config, Trainer.Config):
        converter = _CONFIG_CONVERTERS.get(type(config))
        if converter is not None:
            config = converter.convert(config)
    return config, filtered_args


def apply() -> None:
    """Install the loader wrapper that dispatches registered config converters."""
    ConfigManager._load_config = _patched_load_config  # type: ignore[method-assign]


apply()
