# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim for Qwen3-Next Gated DeltaNet's chunk_gated_delta_rule.

Records the real production op name (fla.ops.gated_delta_rule.
chunk_gated_delta_rule) into the active OpDispatchCapture with
analytically-derived shapes — never invoking the pure-torch chunked
fallback. Mirrors the kda_shim.py pattern.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.qwen3_next.gated_deltanet import GatedDeltaNet
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

_GDN_SHIM_MARKER = "_simulator_gdn_shim_installed"

_CHUNK_GATED_DELTA_RULE_OP = "fla.ops.gated_delta_rule.chunk_gated_delta_rule"
_CHUNK_GATED_DELTA_RULE_GRAD_OP = "fla.ops.gated_delta_rule.chunk_gated_delta_rule_grad"


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


class _SimChunkGatedDeltaRuleFn(torch.autograd.Function):
    """Shape-only autograd bridge for chunk_gated_delta_rule on meta tensors."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, module_path):
        output = run_synthetic_op(
            _CHUNK_GATED_DELTA_RULE_OP,
            inputs=[q, k, v, g, beta, A_log, dt_bias],
            output_factory=lambda: _uncaptured_empty_like(v),
            module_path=module_path,
        )
        ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):
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
            _CHUNK_GATED_DELTA_RULE_GRAD_OP,
            [q, k, v, g, beta, A_log, dt_bias, grad_output],
            grads,
            ctx.module_path,
        )
        return (*grads, None)  # module_path


def sim_chunk_gated_delta_rule(module, q, k, v, g, beta):
    """Record GDN without replacing the already-parallelized module."""
    module_path = _current_module_path()
    A_log = module.A_log.to_local() if isinstance(module.A_log, DTensor) else module.A_log
    dt_bias = module.dt_bias.to_local() if isinstance(module.dt_bias, DTensor) else module.dt_bias
    return _SimChunkGatedDeltaRuleFn.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        module_path,
    )


def apply_qwen3_next_shims(model) -> None:
    """Bind the GDN shape-only shim while preserving module hooks."""
    from types import MethodType

    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            if not getattr(module, _GDN_SHIM_MARKER, False):
                module._chunk_gated_delta_rule = MethodType(
                    sim_chunk_gated_delta_rule, module
                )
                setattr(module, _GDN_SHIM_MARKER, True)
