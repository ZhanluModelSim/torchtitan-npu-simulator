# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install Qwen3-Next shape-only kernel bindings in simulator mode.

Binds the same virtual fused operators used by Kimi K3 (``fusion_attention``
and ``situ_glu``) so the captured workload matches the NPU fused-kernel
execution model (cf. MindSpeed ``torch_npu.npu_fusion_attention`` and fused
SwiGLU) instead of the decomposed SDPA/elementwise chain.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.qwen3_next.attention import GatedAttention
from torchtitan_npu.models.qwen3_next.feed_forward import (
    Qwen3NextGroupedExperts,
    _SharedExpertMLP,
)
from torchtitan_npu.simulator.hardware_shims.kimi_k3_fusion_shim import (
    sim_gated_mla_attention,
    sim_kimi_situ_glu,
)

_FA_SHIM_MARKER = "_simulator_qwen_fa_shim_installed"
_GROUPED_EXPERTS_SHIM_MARKER = "_simulator_qwen_grouped_experts_shim_installed"
_MLP_SHIM_MARKER = "_simulator_qwen_mlp_shim_installed"


def _sim_qwen_grouped_experts(
    module: Qwen3NextGroupedExperts,
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


def _sim_qwen_shared_mlp(module: _SharedExpertMLP, x: torch.Tensor) -> torch.Tensor:
    return module.down_proj(sim_kimi_situ_glu(module.gate_proj(x), module.up_proj(x)))


def apply_qwen3_next_fusion_shims(model) -> None:
    """Bind fused attention / GLU virtual ops while preserving module hooks."""
    for module in model.modules():
        if isinstance(module, GatedAttention):
            if not getattr(module, _FA_SHIM_MARKER, False):
                module._attention_core = MethodType(sim_gated_mla_attention, module)
                setattr(module, _FA_SHIM_MARKER, True)
        elif isinstance(module, Qwen3NextGroupedExperts):
            if not getattr(module, _GROUPED_EXPERTS_SHIM_MARKER, False):
                module.forward = MethodType(_sim_qwen_grouped_experts, module)
                setattr(module, _GROUPED_EXPERTS_SHIM_MARKER, True)
        elif isinstance(module, _SharedExpertMLP):
            if not getattr(module, _MLP_SHIM_MARKER, False):
                module.forward = MethodType(_sim_qwen_shared_mlp, module)
                setattr(module, _MLP_SHIM_MARKER, True)
