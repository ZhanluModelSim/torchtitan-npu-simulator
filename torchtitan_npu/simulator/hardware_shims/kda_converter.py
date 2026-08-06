# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install Kimi K3 shape-only kernel bindings in simulator mode."""

from __future__ import annotations

from types import MethodType

import torch

from torchtitan_npu.converters.kernels.kimi_k3_moe import (
    NpuKimiRMSNormGated,
)
from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention
from torchtitan_npu.simulator.hardware_shims.kda_shim import sim_chunk_kda
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import (
    run_meta_rms_norm,
)

_KDA_SHIM_MARKER = "_simulator_kda_shim_installed"
_GATED_RMS_NORM_SHIM_MARKER = "_simulator_gated_rms_norm_shim_installed"


def _sim_gated_rms_norm(
    module: NpuKimiRMSNormGated,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    normalized = run_meta_rms_norm(x, module.weight, module.eps)
    return normalized * torch.sigmoid(gate.float()).to(x.dtype)


def apply_kimi_k3_shims(model) -> None:
    """Bind KDA and gated RMSNorm shims while preserving module hooks."""
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            if not getattr(module, _KDA_SHIM_MARKER, False):
                module._chunk_kda = MethodType(sim_chunk_kda, module)
                setattr(module, _KDA_SHIM_MARKER, True)
        elif isinstance(module, NpuKimiRMSNormGated):
            if not getattr(module, _GATED_RMS_NORM_SHIM_MARKER, False):
                module.forward = MethodType(_sim_gated_rms_norm, module)
                setattr(module, _GATED_RMS_NORM_SHIM_MARKER, True)
