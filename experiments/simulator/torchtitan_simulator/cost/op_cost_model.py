# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator cost annotations: workload facts plus opt-in duration prediction.

The simulator captures no timing.  What a downstream cost model needs from the
capture is the *work* each operator represents, and that work is a fact of the
shapes and dtypes: a matmul over (M, K) x (K, N) is 2*M*N*K FLOPs and at least
(|A| + |B| + |C|) bytes of memory traffic, on any hardware.  This module
annotates those facts onto the workload graph, one op handler at a time.

Annotations are written only by the explicit :func:`annotate_workload_costs`
call -- the default capture and its consumers are unchanged.

On top of the hardware-independent work, :func:`annotate_workload_costs` can
opt into per-op duration prediction by passing a hardware profile from
:mod:`torchtitan_simulator.cost.hardware`.  Two duration paths exist:

- matmul/bmm go through the workspace ``operator-performance-models`` package
  (its ``MatmulAdaptor`` parsing and ``MatmulRoofline`` formula), and
- every other family uses the same roofline composition locally
  (``max(vec, memory) + overhead`` for memory-bound vec kernels,
  ``max(cube, memory) + overhead`` for grouped GEMM) with the traffic rule
  "each logical input read once, each output written once".

Duration estimates are order-of-magnitude numbers written as ``duration_us``.
Ops that no handler claims stay ``cost_unknown`` rather than guessed; the
per-family equivalent-op counts below are declared modeling assumptions, not
calibrated constants.
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
_OVERHEAD_US = 3.0
_MATMUL_OVERHEAD_US = _OVERHEAD_US


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The hardware-independent work an operator represents.

    ``flops`` counts multiply-accumulate pairs as two FLOPs (cube families
    only).  ``traffic_bytes`` is the least memory the operator must move:
    every tensor it reads plus every tensor it writes, once.  Real caches,
    fusion and layout choices can only reduce the *rate* at which that
    traffic is paid, not the accounting.  ``comm_bytes`` applies to
    collectives only.  ``duration_us`` is the optional roofline prediction
    and is present only when a hardware profile was supplied.
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


def _traffic(tensors: list[TensorMeta]) -> int:
    return sum(tensor_volume_bytes(t.shape, t.dtype) for t in tensors)


def _elements(tensors: list[TensorMeta]) -> int:
    return sum(_numel(t.shape) for t in tensors)


# ---------------------------------------------------------------------------
# Duration paths


def _library_matmul_duration(name: str, inputs: list[TensorMeta], outputs: list[TensorMeta], hardware) -> float | None:
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

        features = OpFeatures(
            raw_name=f"aten.{name.lower()}.default",
            name=name,
            inputs=tuple(TensorFeatures(shape=t.shape, dtype=t.dtype, device=t.device) for t in inputs),
            outputs=tuple(TensorFeatures(shape=t.shape, dtype=t.dtype, device=t.device) for t in outputs),
        )
        parsed = MatmulAdaptor().parse(features)
        estimate = MatmulRoofline(overhead_us=_OVERHEAD_US).predict(parsed, hardware)
        return estimate.duration_us
    except Exception:
        return None


def _cube_duration(hardware, flops: int, traffic: int) -> float | None:
    """``max(cube, memory) + overhead``, mirroring the package composition."""
    if hardware is None or hardware.chip is None:
        return None
    cube = hardware.chip.cube
    hbm = hardware.chip.hbm
    cube_us = flops / cube.effective_flops_per_second(16, flops) * 1e6 if flops else 0.0
    memory_us = traffic / hbm.effective_bytes_per_second(traffic) * 1e6 if traffic else 0.0
    return max(cube_us, memory_us) + _OVERHEAD_US


def _vec_duration(hardware, elements: int, equiv_ops: float, traffic: int) -> float | None:
    """``max(vec, memory) + overhead`` for memory-bound vec kernels.

    ``equiv_ops`` is the declared per-element equivalent-operation count; with
    the counts used here every family stays memory-bound on this chip, which
    keeps the uncalibrated vec peak out of the result.
    """
    if hardware is None or hardware.chip is None:
        return None
    vec_flops = elements * equiv_ops
    vec = hardware.chip.vec
    hbm = hardware.chip.hbm
    vec_us = vec_flops / vec.effective_flops_per_second(16, vec_flops) * 1e6 if vec_flops else 0.0
    memory_us = traffic / hbm.effective_bytes_per_second(traffic) * 1e6 if traffic else 0.0
    return max(vec_us, memory_us) + _OVERHEAD_US


# ---------------------------------------------------------------------------
# Handlers.  Traffic rule throughout: each logical input read once, each
# output written once, at its own dtype width; broadcast inputs are not
# materialized.


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
    traffic = _traffic([a, b, out])
    return CostEstimate(
        flops=flops,
        traffic_bytes=traffic,
        duration_us=_library_matmul_duration("Mm", inputs, outputs, hardware),
    )


def _bmm_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """``aten.bmm``: (B, M, K) @ (B, K, N) -> (B, M, N), equal strict batches."""
    if len(inputs) != 2 or not outputs:
        return CostEstimate.unknown_cost()
    a, b, out = inputs[0], inputs[1], outputs[0]
    if len(a.shape) != 3 or len(b.shape) != 3 or len(out.shape) != 3:
        return CostEstimate.unknown_cost()
    if a.shape[0] != b.shape[0] or out.shape != (a.shape[0], a.shape[1], b.shape[2]):
        return CostEstimate.unknown_cost()
    if a.shape[2] != b.shape[1]:
        return CostEstimate.unknown_cost()
    batch, m, k = a.shape
    n = b.shape[2]
    flops = 2 * batch * m * n * k
    traffic = _traffic([a, b, out])
    return CostEstimate(
        flops=flops,
        traffic_bytes=traffic,
        duration_us=_library_matmul_duration("Bmm", inputs, outputs, hardware),
    )


def _grouped_mm_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """``aten._grouped_mm``: x (T, K) x w (G, K, N) + offsets -> (T, N).

    All groups share K and N; the token rows are partitioned by the offsets
    tensor, so the total FLOPs are ``2 * T * N * K`` regardless of the group
    boundaries (an unbalanced split changes scheduling, not work).
    """
    if len(inputs) < 2 or not outputs:
        return CostEstimate.unknown_cost()
    x, w, out = inputs[0], inputs[1], outputs[0]
    if len(x.shape) != 2 or len(w.shape) != 3 or len(out.shape) != 2:
        return CostEstimate.unknown_cost()
    t, k = x.shape
    if w.shape[1:] != (k, out.shape[1]) or out.shape[0] != t:
        return CostEstimate.unknown_cost()
    flops = 2 * t * out.shape[1] * k
    traffic = _traffic(list(inputs) + list(outputs))
    return CostEstimate(
        flops=flops,
        traffic_bytes=traffic,
        duration_us=_cube_duration(hardware, flops, traffic),
    )


def _pointwise_cost(
    equiv_ops: float,
) -> Callable[[list[TensorMeta], list[TensorMeta], object], CostEstimate]:
    """Memory-bound elementwise family: declared equiv ops per output element."""

    def handler(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
        if not outputs:
            return CostEstimate.unknown_cost()
        traffic = _traffic(list(inputs) + list(outputs))
        elements = _elements(outputs)
        if not traffic or not elements:
            return CostEstimate.unknown_cost()
        return CostEstimate(
            traffic_bytes=traffic,
            duration_us=_vec_duration(hardware, elements, equiv_ops, traffic),
        )

    return handler


def _copy_cost(write_only: bool = False):
    """Pure data movement (cat/split/copy): traffic only, no equivalent work."""

    def handler(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
        if not outputs:
            return CostEstimate.unknown_cost()
        sources = [] if write_only else list(inputs)
        traffic = _traffic(sources + list(outputs))
        if not traffic:
            return CostEstimate.unknown_cost()
        return CostEstimate(
            traffic_bytes=traffic,
            duration_us=_vec_duration(hardware, 0, 0, traffic),
        )

    return handler


_HANDLERS: dict[str, Callable[[list[TensorMeta], list[TensorMeta], object], CostEstimate]] = {
    # Dispatch nodes carry the normalized ``op_type`` ("matmul") alongside the
    # raw ``aten.mm.default`` annotation; the registry keys on the former.
    "matmul": _matmul_cost,
    "bmm": _bmm_cost,
    "grouped_mm": _grouped_mm_cost,
    # Equivalent ops per output element: declared modeling choices (the vec
    # peak is a placeholder; every family here is memory-bound at these counts).
    "rms_norm": _pointwise_cost(4),
    "rms_norm_backward": _pointwise_cost(4),
    "rope": _pointwise_cost(2),
    "silu": _pointwise_cost(4),
    "softmax": _pointwise_cost(8),
    "cat": _copy_cost(),
    "split": _copy_cost(),
    "moe_token_permute": _copy_cost(),
    # Fused AdamW list op: per parameter it reads param/grad/m/v and writes
    # param/m/v(+aux), so traffic dominates; ~10 vector ops per element.
    "adamw_step": _pointwise_cost(10),
}

# Unknown dispatch nodes fall back to their raw op name (``aten.*`` / ``npu.*``
# prefixes and ``.default`` suffixes stripped, overload suffixes kept).  Same
# declared-equivalent-ops convention as above.
_RAW_EQUIV: dict[str, float] = {
    # unary / activation
    "neg": 1,
    "exp": 2,
    "rsqrt": 2,
    "sigmoid": 4,
    "sigmoid_backward": 2,
    "silu_backward": 4,
    "softplus": 4,
    "clamp": 1,
    "pow.Tensor_Scalar": 1,
    "npu_rotary_mul_backward": 3,
    # binary elementwise
    "add.Tensor": 1,
    "add_.Tensor": 1,
    "sub.Tensor": 1,
    "mul.Tensor": 1,
    "mul_.Tensor": 1,
    "div.Tensor": 1,
    "mul.Scalar": 1,
    "div.Scalar": 1,
    # reductions
    "sum": 2,
    "sum.dim_IntList": 2,
    "mean": 2,
    "mean.dim": 2,
    "linalg_vector_norm": 2,
    "amax": 2,
    # softmax family (multi-pass: max/sum/normalize + exp absorbed in equiv)
    "_log_softmax": 8,
    "_log_softmax_backward_data": 8,
    # select / masked
    "where.self": 1,
    "masked_fill.Scalar": 1,
    # comparisons produce bool outputs; traffic carries the cost
    "lt.Scalar": 1,
    "le.Scalar": 1,
    "gt.Scalar": 1,
    "ge.Scalar": 1,
    "eq.Tensor": 1,
    "bitwise_and.Tensor": 1,
    "bitwise_not": 1,
    "logical_and_": 1,
}

_RAW_COPY = {
    "clone",
    "_to_copy",
    "copy_",
    "gather",
    "index_select",
    "index_put_",
    "constant_pad_nd",
    "slice_backward",
    "select_backward",
    "npu_moe_token_unpermute",
    "unpermute",
}

_RAW_WRITE_ONLY = {"zeros", "zeros_like", "new_zeros", "ones_like", "arange"}


def _normalize_raw(raw: str) -> str:
    name = raw
    for prefix in ("aten.", "npu.", "cann_ops_transformer."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.removesuffix(".default")


def _raw_cost(raw: str, inputs: list[TensorMeta], outputs: list[TensorMeta], hardware) -> CostEstimate | None:
    name = _normalize_raw(raw)
    equiv = _RAW_EQUIV.get(name)
    if equiv is not None:
        return _pointwise_cost(equiv)(inputs, outputs, hardware)
    if name in _RAW_COPY:
        return _copy_cost()(inputs, outputs, hardware)
    if name in _RAW_WRITE_ONLY:
        return _copy_cost(write_only=True)(inputs, outputs, hardware)
    return None


def compute_cost(
    op_type: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware=None,
    raw_op_type: str | None = None,
) -> CostEstimate:
    """Look up the handler for one dispatch op; unknown ops stay unknown.

    Passing a hardware profile additionally asks handlers for a duration
    prediction; the default ``None`` keeps every call hardware-independent.
    Nodes dispatching as ``unknown`` fall back to their ``raw_op_type``.
    """
    handler = _HANDLERS.get(op_type)
    if handler is not None:
        try:
            return handler(inputs, outputs, hardware)
        except (IndexError, TypeError, ValueError):
            return CostEstimate.unknown_cost()
    if op_type == "unknown" and raw_op_type:
        try:
            estimate = _raw_cost(raw_op_type, inputs, outputs, hardware)
        except (IndexError, TypeError, ValueError):
            return CostEstimate.unknown_cost()
        if estimate is not None:
            return estimate
    return CostEstimate.unknown_cost()


def _annotate_node(node: OpNode, hardware) -> None:
    cost = compute_cost(node.op_type, node.inputs, node.outputs, hardware, node.annotations.get("raw_op_type"))
    if cost.unknown or (not cost.flops and not cost.traffic_bytes and cost.duration_us is None):
        node.annotations[_UNKNOWN_KEY] = True
        return
    if cost.flops:
        node.annotations[_FLOPS_KEY] = cost.flops
    if cost.traffic_bytes:
        node.annotations[_TRAFFIC_KEY] = cost.traffic_bytes
    if cost.comm_bytes:
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
