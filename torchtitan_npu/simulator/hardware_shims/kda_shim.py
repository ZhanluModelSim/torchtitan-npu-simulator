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

from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


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
        output = torch.empty_like(v)
        _record(
            "triton_ascend_kernels.chunk_kda",
            [q, k, v, g, beta, A_log, dt_bias],
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, g, beta = ctx.saved_tensors
        _record(
            "triton_ascend_kernels.chunk_kda_grad",
            [q, k, v, g, beta, grad_output],
            [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)],
            ctx.module_path,
        )
        return (
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(g),
            torch.empty_like(beta),
            None,  # A_log
            None,  # dt_bias
            None,  # module_path
        )


class SimKimiDeltaAttention(KimiDeltaAttention):
    """Drop-in simulator replacement for KimiDeltaAttention.

    Never runs the real triton chunk_kda kernel; only records the real op name
    + analytically-correct shapes into the capture graph.
    """

    def __init__(self, parent: KimiDeltaAttention) -> None:
        self.__dict__.update(parent.__dict__)

    def _chunk_kda(self, q, k, v, g, beta):
        module_path = _current_module_path()
        return _SimChunkKDAFn.apply(q, k, v, g, beta, self.A_log, self.dt_bias, module_path)
