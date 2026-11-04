# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator cost annotations: hardware-independent FLOPs and memory traffic.

The simulator captures no timing.  What a downstream cost model needs from the
capture is the *work* each operator represents, and that work is a fact of the
shapes and dtypes: a matmul over (M, K) x (K, N) is 2*M*N*K FLOPs and at least
(|A| + |B| + |C|) bytes of memory traffic, on any hardware.  This module
annotates those facts onto the workload graph, one op handler at a time; an
operator without a handler is marked ``cost_unknown`` and left at zero rather
than guessed.

Annotations are written only by the explicit :func:`annotate_workload_costs`
call -- the default capture and its consumers are unchanged.

The numbers follow the previous simulator's conventions (``2 * numel(out) * K``
for matmul, per-tensor byte volumes); they are workload facts, not a latency
model: hardware peak rates, utilization and scheduling stay downstream.
"""

from collections.abc import Callable
from dataclasses import dataclass

from torchtitan_simulator.capture.tensor_utils import tensor_volume_bytes
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.tensor_meta import TensorMeta

__all__ = ["CostEstimate", "annotate_workload_costs", "compute_cost"]

_FLOPS_KEY = "flops"
_TRAFFIC_KEY = "memory_traffic_bytes"
_COMM_KEY = "comm_bytes"
_UNKNOWN_KEY = "cost_unknown"


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The hardware-independent work an operator represents.

    ``flops`` counts multiply-accumulate pairs as two FLOPs.  ``traffic_bytes``
    is the least memory the operator must move: every tensor it reads plus
    every tensor it writes, once.  Real caches, fusion and layout choices can
    only reduce the *rate* at which that traffic is paid, not the accounting.
    ``comm_bytes`` applies to collectives only.
    """

    flops: int = 0
    traffic_bytes: int = 0
    comm_bytes: int = 0
    unknown: bool = False

    @classmethod
    def unknown_cost(cls) -> "CostEstimate":
        return cls(unknown=True)


def _numel(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _matmul_cost(inputs: list[TensorMeta], outputs: list[TensorMeta]) -> CostEstimate:
    """``aten.mm``: (M, K) @ (K, N) -> (M, N), no broadcasting.

    FLOPs are ``2 * M * N * K``: each output element is a K-length dot product
    (one multiply plus one add per element).  The read of A and B and the write
    of C are each counted once at their own dtype width -- with mixed-precision
    capture the inputs are BF16 while an FP32 accumulate output doubles C.
    """
    if len(inputs) != 2 or not outputs:
        return CostEstimate.unknown_cost()
    a, b, out = inputs[0], inputs[1], outputs[0]
    if len(a.shape) != 2 or len(b.shape) != 2 or len(out.shape) != 2:
        return CostEstimate.unknown_cost()
    if a.shape[1] != b.shape[0] or out.shape != (a.shape[0], b.shape[1]):
        return CostEstimate.unknown_cost()
    m, k = a.shape
    n = b.shape[1]
    flops = 2 * m * n * k
    traffic = sum(tensor_volume_bytes(tensor.shape, tensor.dtype) for tensor in (a, b, out))
    return CostEstimate(flops=flops, traffic_bytes=traffic)


_HANDLERS: dict[str, Callable[[list[TensorMeta], list[TensorMeta]], CostEstimate]] = {
    # Dispatch nodes carry the normalized ``op_type`` ("matmul") alongside the
    # raw ``aten.mm.default`` annotation; the registry keys on the former.
    "matmul": _matmul_cost,
}


def compute_cost(op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta]) -> CostEstimate:
    """Look up the handler for one dispatch op; unknown ops stay unknown."""
    handler = _HANDLERS.get(op_type)
    if handler is None:
        return CostEstimate.unknown_cost()
    try:
        return handler(inputs, outputs)
    except (IndexError, TypeError, ValueError):
        return CostEstimate.unknown_cost()


def _annotate_node(node: OpNode) -> None:
    cost = compute_cost(node.op_type, node.inputs, node.outputs)
    if cost.unknown:
        node.annotations[_UNKNOWN_KEY] = True
        return
    node.annotations[_FLOPS_KEY] = cost.flops
    node.annotations[_TRAFFIC_KEY] = cost.traffic_bytes
    node.annotations[_COMM_KEY] = cost.comm_bytes


def annotate_workload_costs(workload) -> None:
    """Stamp ``flops`` / ``memory_traffic_bytes`` / ``comm_bytes`` onto every
    node of every step template in place (captured source graph included).

    Explicit and idempotent: ops without a handler get ``cost_unknown`` only.
    """
    for graph in workload.step_templates.values():
        for node in graph.nodes.values():
            _annotate_node(node)
    for graph in workload.captured_workload.step_templates.values():
        for node in graph.nodes.values():
            _annotate_node(node)
