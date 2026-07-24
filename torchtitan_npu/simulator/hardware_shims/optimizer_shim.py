# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-safe optimizer capture helpers.

In real NPU training, torch._fused_adamw_ dispatches to
npu.npu_apply_adam_w (a fused NPU kernel). Under meta simulation,
fused=True raises RuntimeError (meta device not in supported list).

The public optimizer contract is captured from ``Optimizer.param_groups`` at
the optimizer-step boundary. The low-level fused shim then uses the matching
complete group for operator modeling, while preserving the actual filtered
grad/state operands for memory tracking. This matters because AdamW omits
parameters whose gradient is ``None`` before calling ``torch._fused_adamw_``.

The optimizer OpNode uses logical DTensor global shapes so uneven HSDP shards
do not leak into operator modeling. Tensor dependencies and memory events keep
using per-rank local tensors.

See meta_env._patch_fused_adamw_for_meta for installation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch


@dataclass
class _OptimizerParamScope:
    groups: tuple[tuple[torch.Tensor, ...], ...]
    consumed_groups: set[int] = field(default_factory=set)

    def resolve(self, fused_params: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return the smallest unconsumed param group containing this call."""
        if not fused_params:
            return fused_params
        fused_ids = {id(param) for param in fused_params}
        candidates = [
            (len(group), index, group)
            for index, group in enumerate(self.groups)
            if index not in self.consumed_groups
            and fused_ids <= {id(param) for param in group}
        ]
        if not candidates:
            return fused_params
        _, index, group = min(candidates, key=lambda candidate: candidate[0])
        self.consumed_groups.add(index)
        return list(group)


_active_param_scope: ContextVar[_OptimizerParamScope | None] = ContextVar(
    "simulator_optimizer_param_scope",
    default=None,
)


def _optimizer_param_groups(optimizer_step: Callable) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Read optimizer-owned trainable parameters through public attributes."""
    owner = getattr(optimizer_step, "__self__", None)
    if owner is None:
        return ()

    nested = getattr(owner, "optimizers", None)
    optimizers = tuple(nested) if nested is not None else (owner,)
    groups: list[tuple[torch.Tensor, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for optimizer in optimizers:
        for param_group in getattr(optimizer, "param_groups", ()):
            params = tuple(
                param
                for param in param_group.get("params", ())
                if isinstance(param, torch.Tensor) and param.requires_grad
            )
            key = tuple(id(param) for param in params)
            if params and key not in seen:
                groups.append(params)
                seen.add(key)
    return tuple(groups)


@contextmanager
def capture_optimizer_param_groups(optimizer_step: Callable) -> Iterator[None]:
    """Expose complete public param groups to low-level optimizer shims."""
    groups = _optimizer_param_groups(optimizer_step)
    if not groups:
        yield
        return
    token = _active_param_scope.set(_OptimizerParamScope(groups))
    try:
        yield
    finally:
        _active_param_scope.reset(token)


def _meta_safe_fused_adamw(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
    state_steps, *, amsgrad, beta1, beta2, lr,
    weight_decay, eps, maximize, **_kwargs,
):
    """Replacement for torch._fused_adamw_ under meta simulation.

    Record the NPU fused op name without executing numerical optimizer math.
    """
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    cap = get_active_capture()
    scope = _active_param_scope.get()
    logical_params = scope.resolve(params) if scope is not None else params

    # Model optimizer work from parameter volume only. Gradients and optimizer
    # states are algorithm-derived multiples of that volume; expanding them
    # here makes otherwise identical operands ambiguous to downstream consumers.
    if cap is not None:
        memory_inputs = [
            *params,
            *grads,
            *exp_avgs,
            *exp_avg_sqs,
            *max_exp_avg_sqs,
            *state_steps,
        ]
        memory_outputs = [
            *params,
            *exp_avgs,
            *exp_avg_sqs,
            *max_exp_avg_sqs,
            *state_steps,
        ]
        cap.record_synthetic_op(
            "npu.npu_apply_adam_w.default",
            logical_params or [torch.empty(1, device="meta")],
            logical_params,
            logical_dtensor_shapes=True,
            memory_inputs=memory_inputs,
            memory_outputs=memory_outputs,
        )
