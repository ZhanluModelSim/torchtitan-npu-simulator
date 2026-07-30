# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Installer for Kimi K3 MoE hardware shims in simulator mode.

Handles two cases:
1. NpuKimiK3MoeBlock (converter applied): patches _moe_forward to use
   _SimNpuKimiK3MoeFn instead of torch._grouped_mm.
2. KimiSparseMoeBlock (no converter): patches _moe_forward to a meta-safe
   version that avoids torch.nonzero (unsupported on meta device).
"""

from __future__ import annotations

import torch

from torchtitan_npu.models.kimi_k3.feed_forward import KimiSparseMoeBlock
from torchtitan_npu.simulator.hardware_shims.kimi_k3_moe_shim import sim_npu_kimi_k3_moe_forward


def _make_sim_moe_forward_npu(module):
    """Shape-only forward for NpuKimiK3MoeBlock (3D tensor weights)."""

    def _sim_moe_forward(x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        return sim_npu_kimi_k3_moe_forward(
            module.npu_experts.gate_up_proj,
            module.npu_experts.down_proj,
            x,
            topk_idx,
            topk_weight,
        )

    return _sim_moe_forward


def _make_sim_moe_forward_base(module: KimiSparseMoeBlock):
    """Shape-only forward for base KimiSparseMoeBlock (per-expert ModuleList).

    Avoids torch.nonzero (unsupported on meta) by using first expert only.
    """

    def _sim_moe_forward(x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        # Use first expert for shape inference; scale by routing weight sum
        expert_out = module.experts[0](x)
        scale = topk_weight.sum(dim=-1, keepdim=True) / module.top_k
        return expert_out * scale

    return _sim_moe_forward


def apply_kimi_k3_moe_shims(model) -> None:
    """Patch all Kimi K3 MoE blocks to use shape-only forward on meta device."""
    try:
        from torchtitan_npu.converters.kernels.kimi_k3_moe import NpuKimiK3MoeBlock
    except ImportError:
        NpuKimiK3MoeBlock = None

    for _name, module in model.named_modules():
        if NpuKimiK3MoeBlock is not None and isinstance(module, NpuKimiK3MoeBlock):
            module._moe_forward = _make_sim_moe_forward_npu(module)
        elif isinstance(module, KimiSparseMoeBlock):
            module._moe_forward = _make_sim_moe_forward_base(module)
