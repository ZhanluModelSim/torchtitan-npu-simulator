# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim for NpuKimiK3MoeBlock's grouped_mm on meta tensors.

Mirrors grouped_experts_shim.py pattern: provides a shape-only autograd bridge
so the simulator can trace through the NPU fused MoE path without executing
torch._grouped_mm (which has no meta kernel).
"""

from __future__ import annotations

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _uncaptured_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    """Create a shape-only gradient without adding a fabricated L0 op."""
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


class _SimNpuKimiK3MoeFn(torch.autograd.Function):
    """Shape-only autograd bridge for NPU fused MoE (grouped_mm + permute)."""

    @staticmethod
    def forward(ctx, gate_up_proj, down_proj, x, top_k_index, top_k_weight):  # noqa: ANN001
        ctx.save_for_backward(gate_up_proj, down_proj, x)
        # Output shape: same as input x (num_tokens, hidden_size)
        return torch.empty_like(x)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        gate_up_proj, down_proj, x = ctx.saved_tensors
        d_gate_up = _uncaptured_empty_like(gate_up_proj)
        d_down = _uncaptured_empty_like(down_proj)
        d_x = _uncaptured_empty_like(x)
        return d_gate_up, d_down, d_x, None, None


def sim_npu_kimi_k3_moe_forward(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weight: torch.Tensor,
) -> torch.Tensor:
    """Run NPU fused MoE with a simulator-owned shape-only backward."""
    return _SimNpuKimiK3MoeFn.apply(gate_up_proj, down_proj, x, top_k_index, top_k_weight)
