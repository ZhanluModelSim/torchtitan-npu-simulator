# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autograd-aware EP communication bridge for meta simulation."""

from __future__ import annotations

import torch


def _uncaptured_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    from torchtitan_npu.simulator.capture.dispatch_capture import (
        get_active_capture,
    )

    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


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
