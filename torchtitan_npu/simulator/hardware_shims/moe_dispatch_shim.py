# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autograd-aware EP communication bridge for meta simulation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch


_fp8_dispatch_transport_enabled: ContextVar[bool] = ContextVar(
    "fp8_dispatch_transport_enabled",
    default=False,
)


@contextmanager
def fp8_dispatch_transport_context(enabled: bool):
    """Scope the simulator-only FP8 EP-dispatch transport model."""
    token = _fp8_dispatch_transport_enabled.set(enabled)
    try:
        yield
    finally:
        _fp8_dispatch_transport_enabled.reset(token)


def is_fp8_dispatch_transport_enabled() -> bool:
    return _fp8_dispatch_transport_enabled.get()


def _uncaptured_empty_like(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor, **kwargs)
    with capture.suspend_recording():
        return torch.empty_like(tensor, **kwargs)


def _uncaptured_empty(shape, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=dtype, device=device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=dtype, device=device)


def _record_all_to_all(
    tensor: torch.Tensor,
    output: torch.Tensor,
    group: object,
) -> None:
    from torchtitan_npu.simulator import meta_env
    from torchtitan_npu.simulator.capture.comm_events import (
        _record_comm_with_l0,
        get_active_recorder,
    )

    recorder = get_active_recorder()
    if recorder is None:
        return

    meta_env._comm_layer = "L1"
    _record_comm_with_l0(
        recorder,
        "all_to_all",
        group,
        tensor,
        output,
    )


def _record_fp8_payload_and_scale(tensor: torch.Tensor, group: object) -> None:
    payload = _uncaptured_empty_like(tensor, dtype=torch.float8_e4m3fn)
    scale_shape = (*tensor.shape[:-1], 1, (tensor.shape[-1] + 31) // 32)
    scale = _uncaptured_empty(
        scale_shape,
        dtype=torch.uint8,
        device=tensor.device,
    )
    _record_all_to_all(payload, _uncaptured_empty_like(payload), group)
    _record_all_to_all(scale, _uncaptured_empty_like(scale), group)


class _SimAllToAll(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, tensor: torch.Tensor, group: object) -> torch.Tensor:
        ctx.group = group
        output = _uncaptured_empty_like(tensor)
        _record_all_to_all(tensor, output, group)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _uncaptured_empty_like(grad_output)
        _record_all_to_all(grad_output, grad_input, ctx.group)
        return grad_input, None


def run_meta_all_to_all(
    tensor: torch.Tensor,
    group: object,
) -> torch.Tensor:
    """Return a shape-only all-to-all result and replay it in backward."""
    return _SimAllToAll.apply(tensor, group)


class _SimFP8DispatchAllToAll(torch.autograd.Function):
    """Model FP8 payload/scale dispatch while preserving BF16 backward traffic.

    The downstream meta graph remains BF16 because the production direct-FP8
    grouped-MM autograd bridge is not installed yet. The emitted communication
    events nevertheless match the intended first-stage transport: FP8 payload
    plus one E8M0 scale for each 32-wide hidden-dimension block.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, tensor: torch.Tensor, group: object) -> torch.Tensor:
        ctx.group = group
        _record_fp8_payload_and_scale(tensor, group)
        return _uncaptured_empty_like(tensor)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _uncaptured_empty_like(grad_output)
        if is_fp8_dispatch_transport_enabled():
            _record_fp8_payload_and_scale(grad_output, ctx.group)
        else:
            _record_all_to_all(grad_output, grad_input, ctx.group)
        return grad_input, None


def run_meta_fp8_dispatch_all_to_all(
    tensor: torch.Tensor,
    group: object,
) -> torch.Tensor:
    """Model FP8 payload/scale dispatch with an unchanged BF16 backward."""
    return _SimFP8DispatchAllToAll.apply(tensor, group)
