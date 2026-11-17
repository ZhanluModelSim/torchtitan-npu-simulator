# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve native argument names and literal values without reading tensor data."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch

type TensorBinding = int | None | tuple[TensorBinding, ...]


@dataclass(frozen=True, slots=True)
class _SchemaArgument:
    name: str
    kwarg_only: bool
    has_default: bool
    default: Any
    tensor_type: bool


@lru_cache(maxsize=1024)
def _schema_arguments(func: Any) -> tuple[_SchemaArgument, ...] | None:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return None
    return tuple(
        _SchemaArgument(
            argument.name,
            argument.kwarg_only,
            argument.has_default_value(),
            argument.default_value,
            "Tensor" in str(argument.type),
        )
        for argument in schema.arguments
    )


def _tensor_binding(value: Any, occurrences: itertools.count) -> tuple[TensorBinding, bool, bool]:
    """Return structure, tensor presence and whether the native structure is supported."""
    if isinstance(value, torch.Tensor):
        return next(occurrences), True, True
    if isinstance(value, (tuple, list)):
        children = tuple(_tensor_binding(item, occurrences) for item in value)
        return (
            tuple(child[0] for child in children),
            any(child[1] for child in children),
            all(child[2] for child in children),
        )
    if isinstance(value, dict):
        # Dict values are flattened by dispatch_capture, but their key binding
        # is outside the v1 tuple/list protocol. Still advance occurrence IDs.
        children = tuple(_tensor_binding(item, occurrences) for item in value.values())
        return None, any(child[1] for child in children), False
    return None, False, True


def _reduce_op_literal(value: Any) -> str | None:
    from torch.distributed import ReduceOp

    if isinstance(value, torch.ScriptObject):
        if value._type().qualified_name() != "__torch__.torch.classes.c10d.ReduceOp":
            return None
        # Native ReduceOp accessors are absent from torch's Python stubs.
        operation = cast("Any", value).op()
    elif isinstance(value, ReduceOp):
        operation = cast("Any", value).op
    elif isinstance(value, ReduceOp.RedOpType):
        operation = cast("Any", value).value
    else:
        return None

    operation_id = int(operation)
    # This literal carries identity only. The scalar belongs to collective_facts.
    for name in ("SUM", "PRODUCT", "MIN", "MAX", "AVG", "BAND", "BOR", "BXOR", "PREMUL_SUM"):
        if operation_id == int(getattr(ReduceOp, name)):
            return name.lower()
    return f"unsupported:reduce_op:{operation_id}"


def _normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor_argument": True}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        literal = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        return ("__sim_scalar_float_v1__", literal)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"real": _normalize(value.real), "imag": _normalize(value.imag)}
    if (operation := _reduce_op_literal(value)) is not None:
        return operation
    return str(value)


def operator_attributes(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind native schema names to flattened tensor occurrences and actual literals.

    Tensor occurrence indices follow args then kwargs insertion order, exactly
    like capture's tensor inputs. Schema order is used only for named bindings.
    """
    arguments = _schema_arguments(func)
    if arguments is None:
        return {key: _normalize(value) for key, value in kwargs.items() if not isinstance(value, torch.Tensor)}

    by_name = {argument.name: argument for argument in arguments}
    supplied: dict[str, tuple[Any, TensorBinding, bool, bool]] = {}
    occurrences = itertools.count()
    for index, value in enumerate(args):
        if index >= len(arguments) or arguments[index].kwarg_only:
            raise ValueError(f"Too many positional arguments for {func}")
        supplied[arguments[index].name] = (value, *_tensor_binding(value, occurrences))
    for name, value in kwargs.items():
        if name not in by_name or name in supplied:
            raise ValueError(f"Unknown or duplicate argument {name!r} for {func}")
        supplied[name] = (value, *_tensor_binding(value, occurrences))

    attrs: dict[str, Any] = {}
    bindings: list[tuple[str, TensorBinding]] = []
    bindings_complete = True
    for argument in arguments:
        if argument.name in supplied:
            value, structure, has_tensor, supported = supplied[argument.name]
        else:
            if not argument.has_default:
                raise ValueError(f"Missing argument {argument.name!r} for {func}")
            value = argument.default
            structure, has_tensor, supported = _tensor_binding(value, occurrences)
        if has_tensor and not supported:
            bindings_complete = False
        if supported and (has_tensor or (argument.tensor_type and (value is None or isinstance(value, (tuple, list))))):
            bindings.append((argument.name, structure))
        # A Tensor-declared slot can carry a scalar in real dispatch (for
        # example add.Tensor's other). Its runtime literal must survive too.
        if not argument.tensor_type or (not has_tensor and isinstance(value, (str, int, float, complex, bool))):
            attrs[argument.name] = _normalize(value)
    if bindings_complete:
        attrs["input_tensor_bindings"] = tuple(bindings)
    return attrs
