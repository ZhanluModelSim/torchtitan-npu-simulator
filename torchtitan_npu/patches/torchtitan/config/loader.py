# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/6290

"""Patch ``ConfigLoader.load`` so ``--print-config`` prints compact folded JSON.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

import torchtitan.config.loader
from torchtitan.config.override import OverrideConfig, parse_cli_imports

from torchtitan_npu.patches.torchtitan.config.format import format_config

if TYPE_CHECKING:
    from torchtitan.config.configurable import Configurable

logger = logging.getLogger(__name__)

_original_load = torchtitan.config.loader.ConfigLoader.load


def patched_load(
    self: torchtitan.config.loader.ConfigLoader,
    args: list[str] | None = None,
) -> Configurable.Config:
    """Delegate normal loads to upstream and compact ``--print-config`` output."""
    namespace = self._parser().parse_args(args)
    if not namespace.print_config:
        return _original_load(self, args)

    config = self._load_config(namespace.module, namespace.config)
    if namespace.override:
        override_config = getattr(config, "override", None)
        if not isinstance(override_config, OverrideConfig):
            raise ValueError(
                "--override requires a recipe whose top-level config has an "
                f"`override` field; {type(config).__qualname__} has none."
            )
        override_config.imports.extend(parse_cli_imports(namespace.override))

    self._apply_launch_options(config, namespace)
    self._validate_assets_path(config)

    print(format_config(config))
    raise SystemExit(0)


def apply() -> None:
    if importlib.util.find_spec("torchtitan.config.format") is not None:
        return
    logger.info("[PATCH] ConfigLoader.load --print-config -> format_config (upstream #6290)")
    torchtitan.config.loader.ConfigLoader.load = patched_load  # type: ignore[method-assign]


apply()
