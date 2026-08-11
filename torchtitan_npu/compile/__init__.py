# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time extensions for NPU models."""

__all__ = ["PatternReplacement", "register_pre_aot_patterns"]

import importlib
import os

from .pattern_replacement import PatternReplacement, register_pre_aot_patterns

for module_path in os.environ.get("TORCHTITAN_NPU_PATTERN_IMPORTS", "").split(","):
    if module_path := module_path.strip():
        importlib.import_module(module_path)
