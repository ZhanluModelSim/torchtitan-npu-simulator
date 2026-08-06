# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only autograd bridge for the fused NPU RMSNorm converter."""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.converters.kernels.rms_norm import NPURMSNorm
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _empty(shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=reference.dtype, device=reference.device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=reference.dtype, device=reference.device)


def _empty_like(reference: torch.Tensor) -> torch.Tensor:
    return _empty(tuple(reference.shape), reference)


def _current_module_path() -> str:
    capture = get_active_capture()
    if capture is None or capture.module_path_tracker is None:
        return ""
    return capture.module_path_tracker.current_path()


def _record(
    raw_op_type: str,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    module_path: str,
) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type,
            inputs=inputs,
            outputs=outputs,
            module_path=module_path,
        )


class _SimRMSNorm(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight, epsilon, module_path):
        output = _empty_like(x)
        rstd = _empty((*x.shape[:-1], 1), x)
        _record(
            "npu.npu_rms_norm.default",
            [x, weight],
            [output, rstd],
            module_path,
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.module_path = module_path
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        grad_x = _empty_like(x)
        grad_weight = _empty_like(weight)
        _record(
            "npu.npu_rms_norm_backward.default",
            [grad_output, x, weight, rstd],
            [grad_x, grad_weight],
            ctx.module_path,
        )
        return grad_x, grad_weight, None, None


class SimRMSNorm(NPURMSNorm):
    """Simulator replacement preserving the fused RMSNorm operator boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resolved_eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return run_meta_rms_norm(x, self.weight, resolved_eps)


def run_meta_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Run the simulator-owned fused RMSNorm autograd boundary."""
    if not isinstance(x, DTensor):
        return _SimRMSNorm.apply(
            x,
            weight,
            epsilon,
            _current_module_path(),
        )

    local_weight = weight.to_local() if isinstance(weight, DTensor) else weight
    local_output = _SimRMSNorm.apply(
        x.to_local(),
        local_weight,
        epsilon,
        _current_module_path(),
    )
    return DTensor.from_local(
        local_output,
        x.device_mesh,
        x.placements,
        run_check=False,
        shape=x.shape,
        stride=x.stride(),
    )
