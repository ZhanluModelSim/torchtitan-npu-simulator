# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim for KDA's chunk_kda triton kernel.

Records the real production op name (triton_ascend_kernels.chunk_kda) into the
active OpDispatchCapture with analytically-derived shapes — never invoking the
real triton-ascend-kernels extension. Mirrors smla_shim.py pattern.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


def _uncaptured_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def _current_module_path() -> str:
    capture = get_active_capture()
    if capture is not None and capture.module_path_tracker is not None:
        return capture.module_path_tracker.current_path()
    return ""


class _SimChunkKDAFn(torch.autograd.Function):
    """Shape-only autograd bridge for chunk_kda on meta tensors."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, module_path):  # noqa: ANN001
        # Output shape: same as v (B, S, H, D)
        output = _uncaptured_empty_like(v)
        _record(
            "triton_ascend_kernels.chunk_kda",
            [q, k, v, g, beta, A_log, dt_bias],
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, g, beta, A_log, dt_bias = ctx.saved_tensors
        grads = [
            _uncaptured_empty_like(q),
            _uncaptured_empty_like(k),
            _uncaptured_empty_like(v),
            _uncaptured_empty_like(g),
            _uncaptured_empty_like(beta),
            _uncaptured_empty_like(A_log),
            _uncaptured_empty_like(dt_bias),
        ]
        _record(
            "triton_ascend_kernels.chunk_kda_grad",
            [q, k, v, g, beta, grad_output],
            grads,
            ctx.module_path,
        )
        return (*grads, None)  # module_path


def sim_chunk_kda(module, q, k, v, g, beta):  # noqa: ANN001
    """Record KDA without replacing the already-parallelized module."""
    module_path = _current_module_path()
    A_log = (
        module.A_log.to_local()
        if isinstance(module.A_log, DTensor)
        else module.A_log
    )
    dt_bias = (
        module.dt_bias.to_local()
        if isinstance(module.dt_bias, DTensor)
        else module.dt_bias
    )
    return _SimChunkKDAFn.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        module_path,
    )
