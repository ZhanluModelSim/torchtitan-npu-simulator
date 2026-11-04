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

On top of the hardware-independent work, :func:`annotate_workload_costs` can
opt into per-op duration prediction by passing a hardware profile from
:mod:`torchtitan_simulator.cost.hardware` (built on the workspace's
``operator-performance-models`` roofline package).  Predicted durations are
order-of-magnitude estimates written as a ``duration_us`` annotation; ops that
the prediction layer cannot parse keep the work annotations and simply carry
no duration.
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
_DURATION_KEY = "duration_us"
_UNKNOWN_KEY = "cost_unknown"
_MATMUL_OVERHEAD_US = 3.0


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The hardware-independent work an operator represents.

    ``flops`` counts multiply-accumulate pairs as two FLOPs.  ``traffic_bytes``
    is the least memory the operator must move: every tensor it reads plus
    every tensor it writes, once.  Real caches, fusion and layout choices can
    only reduce the *rate* at which that traffic is paid, not the accounting.
    ``comm_bytes`` applies to collectives only.  ``duration_us`` is the
    optional roofline prediction and is present only when a hardware profile
    was supplied.
    """

    flops: int = 0
    traffic_bytes: int = 0
    comm_bytes: int = 0
    unknown: bool = False
    duration_us: float | None = None

    @classmethod
    def unknown_cost(cls) -> "CostEstimate":
        return cls(unknown=True)


def _numel(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _matmul_duration(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware) -> float | None:
    """Roofline duration via the workspace ``operator-performance-models``.

    Feeds the captured shapes/dtypes through the package's own matmul adaptor
    (the same parsing the platform uses) and lets ``MatmulRoofline`` decide
    between the cube and memory legs.  Any parse the adaptor rejects (unsupported
    dtype, rank, layout) yields ``None`` -- no duration is invented.
    """
    if hardware is None or hardware.chip is None:
        return None
    try:
        from operator_performance_models.adaptors.matmul import MatmulAdaptor
        from operator_performance_models.core import OpFeatures, TensorFeatures
        from operator_performance_models.models.roofline.matmul import MatmulRoofline

        a, b, out = inputs[0], inputs[1], outputs[0]
        features = OpFeatures(
            raw_name="aten.mm.default",
            name="Mm",
            inputs=(
                TensorFeatures(shape=a.shape, dtype=a.dtype, device=a.device),
                TensorFeatures(shape=b.shape, dtype=b.dtype, device=b.device),
            ),
            outputs=(TensorFeatures(shape=out.shape, dtype=out.dtype, device=out.device),),
        )
        parsed = MatmulAdaptor().parse(features)
        estimate = MatmulRoofline(overhead_us=_MATMUL_OVERHEAD_US).predict(parsed, hardware)
        return estimate.duration_us
    except Exception:
        return None


def _matmul_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
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
    return CostEstimate(
        flops=flops,
        traffic_bytes=traffic,
        duration_us=_matmul_duration(inputs, outputs, hardware),
    )


_HANDLERS: dict[str, Callable[[list[TensorMeta], list[TensorMeta], object], CostEstimate]] = {
    # Dispatch nodes carry the normalized ``op_type`` ("matmul") alongside the
    # raw ``aten.mm.default`` annotation; the registry keys on the former.
    "matmul": _matmul_cost,
}


def compute_cost(op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """Look up the handler for one dispatch op; unknown ops stay unknown.

    Passing a hardware profile additionally asks handlers for a duration
    prediction; the default ``None`` keeps every call hardware-independent.
    """
    handler = _HANDLERS.get(op_type)
    if handler is None:
        return CostEstimate.unknown_cost()
    try:
        return handler(inputs, outputs, hardware)
    except (IndexError, TypeError, ValueError):
        return CostEstimate.unknown_cost()


def _annotate_node(node: OpNode, hardware) -> None:
    cost = compute_cost(node.op_type, node.inputs, node.outputs, hardware)
    if cost.unknown:
        node.annotations[_UNKNOWN_KEY] = True
        return
    node.annotations[_FLOPS_KEY] = cost.flops
    node.annotations[_TRAFFIC_KEY] = cost.traffic_bytes
    node.annotations[_COMM_KEY] = cost.comm_bytes
    if cost.duration_us is not None:
        node.annotations[_DURATION_KEY] = cost.duration_us


def annotate_workload_costs(workload, hardware=None) -> None:
    """Stamp ``flops`` / ``memory_traffic_bytes`` / ``comm_bytes`` onto every
    node of every step template in place (captured source graph included).

    Explicit and idempotent: ops without a handler get ``cost_unknown`` only.
    Passing ``hardware`` (see :mod:`torchtitan_simulator.cost.hardware`) also
    stamps ``duration_us`` on ops the prediction layer supports.
    """
    for graph in workload.step_templates.values():
        for node in graph.nodes.values():
            _annotate_node(node, hardware)
    for graph in workload.captured_workload.step_templates.values():
        for node in graph.nodes.values():
            _annotate_node(node, hardware)
