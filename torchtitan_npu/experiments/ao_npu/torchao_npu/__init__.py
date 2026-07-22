# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Trigger handler registrations via side-effect imports.
from . import wrapper_tensors  # noqa: F401
from .configs import ParamSwapConfig

__all__ = [
    "ParamSwapConfig",
]
