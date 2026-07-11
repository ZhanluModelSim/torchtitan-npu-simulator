# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

__all__ = [
    "ModelCustomConfig",
    "ParallelizePlanUpdater",
    "StateDictUpdater",
    "get_model_converter_config",
    "has_npu_converter",
    "register_model_converter",
    "registry",
]

import importlib
import pkgutil
from pathlib import Path

from .model_custom_interface import (
    ModelCustomConfig,
    ParallelizePlanUpdater,
    StateDictUpdater,
)
from .registry import (
    get_model_converter_config,
    has_npu_converter,
    register_model_converter,
    registry,
)


def _auto_search_converter():
    package_dir = Path(__file__).parent

    for subdir in ["kernels", "features"]:
        subdir_path = package_dir / subdir
        if subdir_path.exists():
            # Import the RoPE kernel first: its module-level patch installs the
            # NPU-safe `_reshape_for_broadcast_complex` onto common.rope. Other
            # kernels (e.g. npu_smla) transitively import deepseek_v32.model, which
            # snapshots that symbol via from-import, so the patch must land first.
            modules = sorted(
                pkgutil.iter_modules([str(subdir_path)]),
                key=lambda m: (m.name != "rope", m.name),
            )
            for _, name, _ in modules:
                importlib.import_module(f".{subdir}.{name}", package=__package__)


_auto_search_converter()
