# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Virtual fused Kimi K3 operators used only by meta simulation capture."""

from __future__ import annotations

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def _empty(shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=reference.dtype, device=reference.device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=reference.dtype, device=reference.device)


def _module_path() -> str:
    capture = get_active_capture()
    if capture is None or capture.module_path_tracker is None:
        return ""
    return capture.module_path_tracker.current_path()


def _record(name: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(name, inputs=inputs, outputs=outputs, module_path=path)


class _SimKimiGatedMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, *args):  # noqa: ANN001
        *parameters, module_path = args
        output = _empty_like(hidden_states)
        ctx.save_for_backward(hidden_states, *parameters)
        ctx.module_path = module_path
        _record("simulator.kimi_gated_mla", [hidden_states, *parameters], [output], module_path)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        saved = ctx.saved_tensors
        grads = [_empty_like(tensor) for tensor in saved]
        _record("simulator.kimi_gated_mla_backward", [*saved, grad_output], grads, ctx.module_path)
        return (*grads, None)


class _SimKimiSiTUGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, module_path):  # noqa: ANN001
        output = _empty(tuple(gate.shape), gate)
        ctx.save_for_backward(gate, up)
        ctx.module_path = module_path
        _record("simulator.kimi_situ_glu", [gate, up], [output], module_path)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = _empty_like(gate), _empty_like(up)
        _record("simulator.kimi_situ_glu_backward", [gate, up, grad_output], [grad_gate, grad_up], ctx.module_path)
        return grad_gate, grad_up, None


def sim_kimi_gated_mla(module, hidden_states, attention_masks=None, positions=None):  # noqa: ANN001
    return _SimKimiGatedMLA.apply(hidden_states, *tuple(module.parameters()), _module_path())


def sim_kimi_situ_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _SimKimiSiTUGLU.apply(gate, up, _module_path())
