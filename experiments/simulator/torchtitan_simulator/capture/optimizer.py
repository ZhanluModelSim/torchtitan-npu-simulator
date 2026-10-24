# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Annotate native optimizer dispatches from public parameter groups."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass
class _OptimizerParamScope:
    groups: tuple[tuple[torch.Tensor, ...], ...]

    def resolve(self, fused_params: list[torch.Tensor]) -> list[torch.Tensor]:
        """Map actual fused operands to logical params without expanding groups."""
        logical_by_id = {}
        for group in self.groups:
            for parameter in group:
                logical_by_id[id(parameter)] = parameter
                if isinstance(local := getattr(parameter, "_local_tensor", None), torch.Tensor):
                    logical_by_id[id(local)] = parameter
        return [logical_by_id.get(id(parameter), parameter) for parameter in fused_params]


_active_param_scope: ContextVar[_OptimizerParamScope | None] = ContextVar(
    "simulator_optimizer_param_scope",
    default=None,
)


def _optimizer_owners(optimizer_step: Callable) -> tuple:
    owner = getattr(optimizer_step, "__self__", None)
    if owner is None:
        return ()
    nested = getattr(owner, "optimizers", None)
    return tuple(nested) if nested is not None else (owner,)


def optimizer_state_values(optimizer_step: Callable):
    """Read actual optimizer state; None means the callable exposes no public owner."""
    from collections.abc import Mapping

    owners = _optimizer_owners(optimizer_step)
    if not owners or any(not isinstance(getattr(owner, "state", None), Mapping) for owner in owners):
        return None
    return [owner.state for owner in owners]


def _optimizer_param_groups(optimizer_step: Callable) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Read optimizer-owned trainable parameters through public attributes."""
    optimizers = _optimizer_owners(optimizer_step)
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
    """Expose complete public param groups to native optimizer dispatch."""
    groups = _optimizer_param_groups(optimizer_step)
    if not groups:
        yield
        return
    token = _active_param_scope.set(_OptimizerParamScope(groups))
    try:
        yield
    finally:
        _active_param_scope.reset(token)


def optimizer_annotations(params):
    scope = _active_param_scope.get()
    logical_params = scope.resolve(params) if scope is not None else params
    return {
        "optimizer_updated_parameter_count": len(logical_params),
        "optimizer_declared_group_sizes": [len(group) for group in scope.groups] if scope is not None else [],
        "optimizer_logical_parameter_shapes": [tuple(parameter.shape) for parameter in logical_params],
        "optimizer_work_scope": "actual_updated_parameters",
    }
