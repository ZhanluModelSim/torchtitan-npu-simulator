# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install Kimi K3 shape-only kernel bindings in simulator mode."""

from __future__ import annotations

from types import MethodType

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.converters.kernels.kimi_k3_moe import (
    NpuKimiRMSNormGated,
)
from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention
from torchtitan_npu.models.kimi_k3.attention import KimiGatedMLA
from torchtitan_npu.models.kimi_k3.feed_forward import KimiGroupedExperts, KimiMLP
from torchtitan_npu.simulator.hardware_shims.kda_shim import sim_chunk_kda
from torchtitan_npu.simulator.hardware_shims.kimi_k3_fusion_shim import (
    sim_kimi_gated_mla,
    sim_kimi_situ_glu,
)
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import (
    run_meta_rms_norm,
)

_KDA_SHIM_MARKER = "_simulator_kda_shim_installed"
_GATED_RMS_NORM_SHIM_MARKER = "_simulator_gated_rms_norm_shim_installed"
_MLA_SHIM_MARKER = "_simulator_mla_shim_installed"
_MLP_SHIM_MARKER = "_simulator_mlp_shim_installed"
_GROUPED_EXPERTS_SHIM_MARKER = "_simulator_grouped_experts_shim_installed"


def _sim_gated_rms_norm(
    module: NpuKimiRMSNormGated,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    normalized = run_meta_rms_norm(x, module.weight, module.eps)
    return normalized * torch.sigmoid(gate.float()).to(x.dtype)


def _sim_kimi_mlp(module: KimiMLP, x: torch.Tensor) -> torch.Tensor:
    gate = module.gate_proj(x)
    up = module.up_proj(x)
    return module.down_proj(sim_kimi_situ_glu(gate, up))


def _sim_kimi_grouped_experts(
    module: KimiGroupedExperts,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module.w1, DTensor):
        w1, w2, w3 = module.w1.to_local(), module.w2.to_local(), module.w3.to_local()
    else:
        w1, w2, w3 = module.w1, module.w2, module.w3

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    gate = torch._grouped_mm(
        x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
    )
    up = torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    return torch._grouped_mm(
        sim_kimi_situ_glu(gate, up),
        w2.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).type_as(x)


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
        elif isinstance(module, KimiGatedMLA):
            if not getattr(module, _MLA_SHIM_MARKER, False):
                module.forward = MethodType(sim_kimi_gated_mla, module)
                setattr(module, _MLA_SHIM_MARKER, True)
        elif isinstance(module, KimiMLP):
            if not getattr(module, _MLP_SHIM_MARKER, False):
                module.forward = MethodType(_sim_kimi_mlp, module)
                setattr(module, _MLP_SHIM_MARKER, True)
        elif isinstance(module, KimiGroupedExperts):
            if not getattr(module, _GROUPED_EXPERTS_SHIM_MARKER, False):
                module.forward = MethodType(_sim_kimi_grouped_experts, module)
                setattr(module, _GROUPED_EXPERTS_SHIM_MARKER, True)
