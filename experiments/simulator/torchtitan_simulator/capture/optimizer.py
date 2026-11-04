# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Annotate native optimizer dispatches from public parameter groups."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True)
class FusedAdamwInputs:
    """Native argument bindings before tensor-list flattening loses their roles."""

    tensor_lists: tuple[tuple[torch.Tensor, ...], ...]
    optional_tensors: tuple[tuple[str, torch.Tensor], ...]

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for group in self.tensor_lists for tensor in group) + tuple(
            tensor for _, tensor in self.optional_tensors
        )

    def attributes(self) -> dict[str, list[int] | list[str]]:
        return {
            "input_tensor_list_sizes": [len(group) for group in self.tensor_lists],
            "optional_tensor_roles": [name for name, _ in self.optional_tensors],
        }


def bind_fused_adamw_inputs(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> FusedAdamwInputs:
    """Bind the supported scalar-LR schema, never infer roles from shape or dtype."""
    list_names = ("self", "grads", "exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs", "state_steps")
    optional_names = ("grad_scale", "found_inf")
    schema = func._schema
    tensor_names = tuple(argument.name for argument in schema.arguments if "Tensor" in str(argument.type))
    if str(func) != "aten._fused_adamw_.default" or tensor_names != list_names + optional_names:
        raise RuntimeError("Unsupported fused AdamW operand schema")
    bound = {
        argument.name: args[index] if index < len(args) else kwargs.get(argument.name, argument.default_value)
        for index, argument in enumerate(schema.arguments)
    }
    groups: list[tuple[torch.Tensor, ...]] = []
    for name in list_names:
        value = bound[name]
        if not isinstance(value, (tuple, list)) or any(not isinstance(item, torch.Tensor) for item in value):
            raise ValueError(f"Fused AdamW {name} must be a tensor list")
        groups.append(tuple(value))
    optional: list[tuple[str, torch.Tensor]] = []
    for name in optional_names:
        value = bound[name]
        if value is not None:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Fused AdamW {name} must be a tensor or None")
            optional.append((name, value))
    return FusedAdamwInputs(tuple(groups), tuple(optional))


@dataclass
class _OptimizerParamScope:
    groups: tuple[tuple[torch.Tensor, ...], ...]
    names_by_id: dict[int, str]

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

_active_parameter_update: ContextVar[tuple[torch.Tensor, str] | None] = ContextVar(
    "simulator_optimizer_parameter_update", default=None
)


@contextmanager
def capture_parameter_update(parameter: torch.Tensor, fqn: str) -> Iterator[None]:
    """Bind a native per-parameter update to its actual mutable operand."""
    token = _active_parameter_update.set((parameter, fqn))
    try:
        yield
    finally:
        _active_parameter_update.reset(token)


def parameter_update_annotations(raw_op_type: str, mutated_inputs: list[torch.Tensor]) -> dict:
    update = _active_parameter_update.get()
    # DistMuon's native _apply_update ends with add_(direction). Its earlier
    # mul_(weight_decay) alone is not evidence of a complete optimizer update.
    if update is None or raw_op_type != "aten.add_.Tensor":
        return {}
    parameter, fqn = update
    local = getattr(parameter, "_local_tensor", parameter)
    storage = local.untyped_storage()._cdata
    if not any(tensor.untyped_storage()._cdata == storage for tensor in mutated_inputs):
        return {}
    return {"optimizer_updated_parameter_names": [fqn], "optimizer_algorithm": "DistMuon"}


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


def optimizer_parameter_manifest(optimizer_step: Callable) -> list[dict]:
    """Read the native mixed container's parameter ownership, without regex inference."""
    return [
        {"algorithm": type(owner).__name__, "parameter_names": list(group.get("param_names", ()))}
        for owner in _optimizer_owners(optimizer_step)
        for group in owner.param_groups
    ]


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
    names_by_id = {
        id(parameter): name
        for owner in _optimizer_owners(optimizer_step)
        for group in owner.param_groups
        if "param_names" in group
        for parameter, name in zip(group["params"], group["param_names"], strict=True)
    }
    token = _active_param_scope.set(_OptimizerParamScope(groups, names_by_id))
    try:
        yield
    finally:
        _active_param_scope.reset(token)


def optimizer_annotations(params):
    scope = _active_param_scope.get()
    logical_params = scope.resolve(params) if scope is not None else params
    annotations: dict[str, Any] = {
        "optimizer_updated_parameter_count": len(logical_params),
        "optimizer_declared_group_sizes": [len(group) for group in scope.groups] if scope is not None else [],
        "optimizer_logical_parameter_shapes": [tuple(parameter.shape) for parameter in logical_params],
        "optimizer_work_scope": "actual_updated_parameters",
    }
    if scope is not None and all(id(parameter) in scope.names_by_id for parameter in logical_params):
        annotations["optimizer_updated_parameter_names"] = [scope.names_by_id[id(p)] for p in logical_params]
        annotations["optimizer_algorithm"] = "AdamW"
    return annotations
