# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maps a canonical L0 op_type + tensor metadata to a CostEstimate.
See design doc §5.8 for the formulas and the rationale for never raising on
an unrecognized op_type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torchtitan_npu.simulator.capture.tensor_utils import tensor_volume_bytes
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


@dataclass
class CostEstimate:
    flops: int = 0
    peak_mem: int = 0
    param_mem: int = 0
    comm_bytes: int = 0
    unknown: bool = False

    @classmethod
    def unknown_cost(cls) -> "CostEstimate":
        return cls(unknown=True)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


class OpCostModel:
    """Registry of `op_type -> handler` cost estimators."""

    def __init__(self) -> None:
        Handler = Callable[[list[TensorMeta], list[TensorMeta], dict[str, Any]], CostEstimate]
        self._handlers: dict[str, Handler] = {
            "matmul": self._matmul,
            "addmm": self._matmul,
            "bmm": self._bmm,
            "grouped_mm": self._matmul,
            "sdpa": self._attention,
            "flash_attention_fwd": self._attention,
            "layer_norm": self._norm,
            "rms_norm": self._norm,
            "gelu": self._elementwise,
            "silu": self._elementwise,
            "swiglu": self._elementwise,
            "softmax": self._elementwise,
            "rope": self._elementwise,
            "moe_token_permute": self._data_move,
            "moe_token_unpermute": self._data_move,
            "moe_re_routing": self._data_move,
            "allreduce": self._allreduce,
            "reduce_scatter": self._allreduce,
            "allgather": self._allgather,
            "all_to_all": self._allgather,
        }

    def compute(
        self,
        op_type: str,
        inputs: list[TensorMeta],
        outputs: list[TensorMeta],
        attrs: dict[str, Any] | None = None,
    ) -> CostEstimate:
        handler = self._handlers.get(op_type)
        if handler is None:
            return CostEstimate.unknown_cost()
        return handler(inputs, outputs, attrs or {})

    def _matmul(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if len(inputs) < 2 or not outputs:
            return CostEstimate.unknown_cost()
        k = inputs[0].shape[-1]
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        param_bytes = tensor_volume_bytes(inputs[1].shape, inputs[1].dtype) if inputs[1].is_parameter else 0
        return CostEstimate(flops=2 * _numel(outputs[0].shape) * k, peak_mem=out_bytes, param_mem=param_bytes)

    def _bmm(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        return self._matmul(inputs, outputs, attrs)

    def _attention(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if len(inputs) < 2 or not outputs:
            return CostEstimate.unknown_cost()
        key_shape = inputs[1].shape
        seq_k = key_shape[-2] if len(key_shape) >= 2 else key_shape[-1]
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(flops=2 * _numel(outputs[0].shape) * seq_k, peak_mem=out_bytes)

    def _norm(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs or not outputs:
            return CostEstimate.unknown_cost()
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(flops=5 * _numel(inputs[0].shape), peak_mem=out_bytes)

    def _elementwise(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs or not outputs:
            return CostEstimate.unknown_cost()
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(flops=_numel(inputs[0].shape), peak_mem=out_bytes)

    def _data_move(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not outputs:
            return CostEstimate.unknown_cost()
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(peak_mem=out_bytes)

    def _allreduce(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs:
            return CostEstimate.unknown_cost()
        total_bytes = tensor_volume_bytes(inputs[0].shape, inputs[0].dtype)
        return CostEstimate(comm_bytes=total_bytes * 2)  # reduce + broadcast

    def _allgather(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs:
            return CostEstimate.unknown_cost()
        return CostEstimate(comm_bytes=tensor_volume_bytes(inputs[0].shape, inputs[0].dtype))
