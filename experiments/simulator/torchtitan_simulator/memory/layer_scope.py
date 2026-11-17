# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared layer identity for saved-activation attribution and reporting."""

import re

_LAYER_PATH_PATTERN = re.compile(r"(?:^|\.)((?:mtp_)?layers\.\d+)(?:\.|$)")


def activation_layer_scope(module_path: str) -> str:
    match = _LAYER_PATH_PATTERN.search(module_path)
    return match.group(1) if match else ""
