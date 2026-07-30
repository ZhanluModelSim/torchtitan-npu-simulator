# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Installer for KDA hardware shims in simulator mode.

Mirrors smla_converter.py pattern: replaces every KimiDeltaAttention with
SimKimiDeltaAttention so the simulator never invokes triton-ascend-kernels.
"""

from __future__ import annotations

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention
from torchtitan_npu.simulator.hardware_shims.kda_shim import SimKimiDeltaAttention


def apply_kda_shims(model) -> None:
    """Replace all KimiDeltaAttention modules with SimKimiDeltaAttention."""
    for name, module in model.named_modules():
        if isinstance(module, KimiDeltaAttention) and not isinstance(module, SimKimiDeltaAttention):
            replace_module_with_name(model, name, SimKimiDeltaAttention(module))
