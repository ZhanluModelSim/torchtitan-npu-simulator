# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Installer for the KDA shape-only kernel binding in simulator mode."""

from __future__ import annotations

from types import MethodType

from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention
from torchtitan_npu.simulator.hardware_shims.kda_shim import sim_chunk_kda

_KDA_SHIM_MARKER = "_simulator_kda_shim_installed"


def apply_kda_shims(model) -> None:
    """Bind the shape-only KDA core while preserving hooks and DTensors."""
    for module in model.modules():
        if not isinstance(module, KimiDeltaAttention):
            continue
        if getattr(module, _KDA_SHIM_MARKER, False):
            continue
        module._chunk_kda = MethodType(sim_chunk_kda, module)
        setattr(module, _KDA_SHIM_MARKER, True)
