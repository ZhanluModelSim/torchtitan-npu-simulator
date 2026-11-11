# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""System3 Flash decoder (DeepSeekV4.1-Flash reference port)."""

from .config_registry import (
    model_registry,
    system3_debugmodel_multimodal,
    system3_debugmodel_text,
)
from .model import System3Model

__all__ = [
    "System3Model",
    "model_registry",
    "system3_debugmodel_multimodal",
    "system3_debugmodel_text",
]
