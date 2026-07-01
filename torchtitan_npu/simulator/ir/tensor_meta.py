# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 tensor metadata: see spec/L0-OpNode.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TensorMeta:
    """Minimal, framework-agnostic description of a single tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    is_parameter: bool = False
