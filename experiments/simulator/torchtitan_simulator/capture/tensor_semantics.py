# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Author-declared tensor roles and semantic axes for simulator-owned operators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchtitan_simulator.ir.tensor_meta import TensorMeta

# A tuple describes components flattened into one axis, not several axes.
type SemanticAxis = str | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class TensorMeaning:
    role: str
    axes: tuple[SemanticAxis, ...]
    unknown_leading_axes: bool = False
    optional: bool = False
    rank_alternatives: tuple[tuple[SemanticAxis, ...], ...] = ()

    def for_tensor(self, tensor: TensorMeta, *, index: int) -> dict[str, Any]:
        rank = len(tensor.shape)
        axes = next((axes for axes in self.rank_alternatives if len(axes) == rank), self.axes)
        if rank < len(axes) or (not self.unknown_leading_axes and rank != len(axes)):
            raise ValueError(f"Tensor role {self.role!r} expects axes {self.axes}, got shape {tensor.shape}")
        axes = (None,) * (rank - len(axes)) + axes
        return {"index": index, "role": self.role, "axes": axes}


@dataclass(frozen=True, slots=True)
class OperatorTensorSemantics:
    inputs: tuple[tuple[str, TensorMeaning], ...]
    outputs: tuple[TensorMeaning, ...]

    def bind(self, attrs: dict[str, Any], inputs: list[TensorMeta], outputs: list[TensorMeta]) -> dict[str, Any]:
        bindings = dict(attrs.get("input_tensor_bindings", ()))
        records = []
        indices = []
        for argument, meaning in self.inputs:
            index = bindings.get(argument)
            if index is None and meaning.optional and argument in bindings:
                continue
            if type(index) is not int or not 0 <= index < len(inputs):
                raise ValueError(f"Custom operator argument {argument!r} needs one captured tensor binding")
            indices.append(index)
            records.append(meaning.for_tensor(inputs[index], index=index))
        if sorted(indices) != list(range(len(inputs))) or len(outputs) != len(self.outputs):
            raise ValueError("Custom operator tensor semantics must describe every input and output exactly once")
        return {
            "version": 1,
            "inputs": tuple(sorted(records, key=lambda record: record["index"])),
            "outputs": tuple(
                meaning.for_tensor(tensor, index=index)
                for index, (meaning, tensor) in enumerate(zip(self.outputs, outputs, strict=True))
            ),
        }


type SemanticsDeclaration = OperatorTensorSemantics | Callable[[dict[str, Any]], OperatorTensorSemantics]

_SEMANTICS: dict[str, SemanticsDeclaration] = {}


def register_tensor_semantics(raw_op_type: str, semantics: SemanticsDeclaration) -> None:
    """Register at the custom operator definition; never match canonical names."""
    if raw_op_type in _SEMANTICS:
        raise ValueError(f"Tensor semantics already registered for {raw_op_type}")
    _SEMANTICS[raw_op_type] = semantics


def attach_tensor_semantics(
    raw_op_type: str, attrs: dict[str, Any], inputs: list[TensorMeta], outputs: list[TensorMeta]
) -> dict[str, Any]:
    """Add only explicitly declared source facts, without inferring from shape."""
    semantics = _SEMANTICS.get(raw_op_type)
    if semantics is None:
        return attrs
    if "tensor_semantics" in attrs:
        raise ValueError(f"Duplicate tensor semantics for {raw_op_type}")
    if callable(semantics):
        semantics = semantics(attrs)
    return {**attrs, "tensor_semantics": semantics.bind(attrs, inputs, outputs)}
