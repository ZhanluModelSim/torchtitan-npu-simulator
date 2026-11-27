# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator cost routing: workload facts plus library-backed duration prediction.

The simulator captures no timing.  What a downstream cost model needs from the
capture is the *work* each operator represents -- flops and least memory
traffic, which are facts of the shapes and dtypes -- and an optional duration
prediction.  **All operator performance modeling lives in the
``operator-performance-models`` package**: the roofline formulas, the declared
per-operator equivalent-work weights, the copy/write-only classes, the mHC
einsum weights and the GDR attention family are owned there.  This module is
routing glue only: it maps each captured node to the library family that
models it and stamps the resulting facts onto the workload graph.

Two routing layers exist:

- dispatch nodes with a dedicated library family (matmul/bmm, grouped GEMM,
  SLA2 fusions, simulator virtual fusions such as ``rms_norm`` / ``rope`` /
  ``mhc_post``), routed by their declared ``(family, raw spelling)`` pair, and
- everything else through the library's :func:`builtin_raw_index`, which
  resolves the raw operator spelling to the precise structural family where
  one exists (matmul, convolution, elementwise, reductions, GDR, ...) and to
  the declared fallback families otherwise.  Ops nothing claims stay
  ``cost_unknown`` rather than guessed.

Annotations are written only by the explicit :func:`annotate_workload_costs`
call -- the default capture and its consumers are unchanged.
"""

from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING

from torchtitan_simulator.capture.tensor_utils import tensor_volume_bytes
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.tensor_meta import TensorMeta

__all__ = ["CostEstimate", "annotate_workload_costs", "compute_cost"]

if TYPE_CHECKING:
    from operator_performance_models.core import Hardware
    from operator_performance_models.models.roofline.base._defaults import SemanticBackend

_FLOPS_KEY = "flops"
_TRAFFIC_KEY = "memory_traffic_bytes"
_COMM_KEY = "comm_bytes"
_DURATION_KEY = "duration_us"
_UNKNOWN_KEY = "cost_unknown"
_OVERHEAD_US = 3.0

_OP_BACKEND: "SemanticBackend | None" = None
_RAW_INDEX: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The hardware-independent work an operator represents.

    ``flops`` counts multiply-accumulate pairs as two FLOPs (cube families
    only).  ``traffic_bytes`` is the least memory the operator must move:
    every tensor it reads plus every tensor it writes, once.  Real caches,
    fusion and layout choices can only reduce the *rate* at which that
    traffic is paid, not the accounting.  ``comm_bytes`` applies to
    collectives only.  ``duration_us`` is the library's roofline prediction
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


# ---------------------------------------------------------------------------
# Library routing.  Every duration prediction (and the facts it reports) comes
# from operator-performance-models; nothing is composed locally.


def _opm_backend() -> "SemanticBackend":
    """The package's builtin roofline backend, built once per process."""
    global _OP_BACKEND
    if _OP_BACKEND is None:
        from operator_performance_models import create_backend

        _OP_BACKEND = create_backend("roofline")
    return _OP_BACKEND


def _raw_index() -> dict[str, str]:
    """Raw spelling -> canonical family name (package-owned index)."""
    global _RAW_INDEX
    if _RAW_INDEX is None:
        from operator_performance_models.adaptors.vec.declared import builtin_raw_index

        _RAW_INDEX = builtin_raw_index()
    return _RAW_INDEX


def _freeze_attr(value):
    """Capture attrs may carry dict/list structures (tensor semantics facts);
    the library accepts only immutable scalars and tuples."""
    if isinstance(value, dict):
        return tuple((k, _freeze_attr(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_attr(v) for v in value)
    return value


def _opm_features(
    canonical: str,
    raw: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    attrs: dict | None = None,
):
    from operator_performance_models.core import OpFeatures, TensorFeatures

    return OpFeatures(
        raw_name=raw,
        name=canonical,
        inputs=tuple(TensorFeatures(shape=t.shape, dtype=t.dtype, device=t.device) for t in inputs),
        outputs=tuple(TensorFeatures(shape=t.shape, dtype=t.dtype, device=t.device) for t in outputs),
        attributes=tuple((k, _freeze_attr(v)) for k, v in (attrs or {}).items()),
    )


def _routed_cost(
    canonical: str,
    raw: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware: "Hardware | None",
    attrs: dict | None = None,
) -> CostEstimate | None:
    """One library prediction: facts and duration straight from the estimate.

    Returns ``None`` (stays unknown) without chip capabilities or when the
    library refuses the geometry -- refusals are explicit upstream, never
    guessed here.
    """
    if hardware is None or hardware.chip is None:
        return None
    try:
        prediction = _opm_backend().predict_detailed(_opm_features(canonical, raw, inputs, outputs, attrs), hardware)
    except Exception:
        return None
    estimate = prediction.estimate
    if estimate.duration_us is None or estimate.details is None:
        return None
    details = estimate.details
    flops = int((details.cube_flops or 0) + (details.vec_flops or 0))
    traffic = int((details.read_bytes or 0) + (details.write_bytes or 0))
    if not flops and not traffic:
        return None
    return CostEstimate(flops=flops, traffic_bytes=traffic, duration_us=estimate.duration_us)


# ---------------------------------------------------------------------------
# Dispatch handlers.


def _matmul_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """``aten.mm``: (M, K) @ (K, N) -> (M, N), routed through the package."""
    estimate = _routed_cost("Mm", "aten.mm.default", inputs, outputs, hardware)
    if estimate is None:
        return CostEstimate.unknown_cost()
    return estimate


def _bmm_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """``aten.bmm``: (B, M, K) @ (B, K, N) -> (B, M, N), routed through the package."""
    estimate = _routed_cost("Bmm", "aten.bmm.default", inputs, outputs, hardware)
    if estimate is None:
        return CostEstimate.unknown_cost()
    return estimate


def _grouped_mm_cost(
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware=None,
    attrs: dict | None = None,
) -> CostEstimate:
    """``aten._grouped_mm``: 2D packed, per-expert 3D and backward weight
    gradients, routed through the package's grouped-matmul family.

    The package prices uneven expert splits only from captured group facts,
    so the capture's ``offs`` value fact (``group_offsets``, cumulative ends)
    is forwarded verbatim; every other captured attribute stays out of the
    grouped-matmul attribute contract.
    """
    facts: dict = {}
    offsets = (attrs or {}).get("group_offsets")
    if offsets:
        try:
            captured = tuple(int(value) for value in offsets)
        except (TypeError, ValueError):
            captured = ()
        if captured and all(type(value) is int for value in captured):
            facts["group_offsets"] = captured
    estimate = _routed_cost("GroupedMm", "aten._grouped_mm.default", inputs, outputs, hardware, facts)
    if estimate is None:
        return CostEstimate.unknown_cost()
    return estimate


def _declared_cost(canonical: str, raw: str):
    """A declared-family route: facts and duration from the package table."""

    def handler(
        inputs: list[TensorMeta],
        outputs: list[TensorMeta],
        hardware=None,
        attrs: dict | None = None,
    ) -> CostEstimate:
        estimate = _routed_cost(canonical, raw, inputs, outputs, hardware, attrs)
        if estimate is None:
            return CostEstimate.unknown_cost()
        return estimate

    return handler


# Simulator dispatch spellings for the virtual fusions and capture-side
# families from upstream workspaces; the package's declared tables carry the
# native aten spellings for the raw fallback path.
_DECLARED_DISPATCH: dict[str, tuple[str, str]] = {
    # op_type -> (declared canonical, raw spelling the package knows)
    "rms_norm": ("Declared.RmsNorm", "rms_norm"),
    "rms_norm_backward": ("Declared.RmsNorm", "rms_norm_backward"),
    "rope": ("Declared.Rope", "rope"),
    "silu": ("Declared.Silu", "silu"),
    "softmax": ("Declared.Softmax", "softmax"),
    "gelu": ("Declared.Gelu", "gelu"),
    "gelu_backward": ("Declared.GeluBackward", "gelu_backward"),
    "adamw_step": ("Declared.AdamwStep", "adamw_step"),
    "cat": ("Declared.Copy", "cat"),
    "split": ("Declared.Copy", "split"),
    "moe_token_permute": ("Declared.Copy", "moe_token_permute"),
    "mhc_post": ("Declared.HcEinsum", "mhc_post"),
    "mhc_post_backward": ("Declared.HcEinsumBackward", "mhc_post_backward"),
}


def _sla2_families() -> dict:
    """Import-guarded adaptor/model table for the four SLA2 library families.

    The structural formulas (cube GEMM populations, declared softmax/normalize
    vector units) live in ``operator-performance-models`` (see its
    ``docs/sla2-roofline-models.md``); this adapter only translates captured
    tensor facts and the ``selected_blocks`` schema scalar.
    """
    from operator_performance_models.adaptors.mix.sla2 import (
        Sla2LinearAttentionAdaptor,
        Sla2LinearAttentionGradientAdaptor,
        Sla2SparseAttentionAdaptor,
        Sla2SparseAttentionGradientAdaptor,
    )
    from operator_performance_models.models.roofline.mix.sla2_linear_attention import (
        Sla2LinearAttentionGradientRoofline,
        Sla2LinearAttentionRoofline,
    )
    from operator_performance_models.models.roofline.mix.sla2_sparse_attention import (
        Sla2SparseAttentionGradientRoofline,
        Sla2SparseAttentionRoofline,
    )

    return {
        "sla2_sparse_attention": (
            "Sla2SparseAttention",
            "fused.sla2_sparse_attention.default",
            Sla2SparseAttentionAdaptor(),
            Sla2SparseAttentionRoofline,
        ),
        "sla2_sparse_attention_gradient": (
            "Sla2SparseAttentionGradient",
            "fused.sla2_sparse_attention_grad.default",
            Sla2SparseAttentionGradientAdaptor(),
            Sla2SparseAttentionGradientRoofline,
        ),
        "sla2_linear_attention": (
            "Sla2LinearAttention",
            "fused.sla2_linear_attention.default",
            Sla2LinearAttentionAdaptor(),
            Sla2LinearAttentionRoofline,
        ),
        "sla2_linear_attention_gradient": (
            "Sla2LinearAttentionGradient",
            "fused.sla2_linear_attention_grad.default",
            Sla2LinearAttentionGradientAdaptor(),
            Sla2LinearAttentionGradientRoofline,
        ),
    }


# Declared vector configuration for the SLA2 fusions: equivalent_ops 1.0 means
# the library's structural unit counts are taken literally, and FP32 compute
# matches the debug model's FP32 parameters. Explicitly declared, not
# calibrated; a consumer may override by extending the cost layer.
_SLA2_VEC_CONFIG_KWARGS = {"compute_bits": 32, "equivalent_ops": 1.0, "compute_dtype": "float32"}


def _sla2_library_cost(
    op_type: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware,
    attrs: dict | None,
) -> CostEstimate:
    """Route an SLA2 fusion to its ``operator-performance-models`` roofline model.

    The library roofline requires chip capabilities, so without a hardware
    profile the nodes stay ``cost_unknown`` rather than partially annotated;
    with one, the estimate carries cube+vec work, logical traffic and a
    duration from the declared coarse policy.
    """
    if hardware is None or hardware.chip is None:
        return CostEstimate.unknown_cost()
    selected = (attrs or {}).get("selected_blocks")
    if type(selected) is not int:
        return CostEstimate.unknown_cost()
    try:
        from operator_performance_models.core import OpFeatures, TensorFeatures
        from operator_performance_models.roofline_config import VecOpConfig

        canonical, raw_name, adaptor, model = _sla2_families()[op_type]
        features = OpFeatures(
            raw_name=raw_name,
            name=canonical,
            inputs=tuple(TensorFeatures(shape=t.shape, dtype=t.dtype, device=t.device) for t in inputs),
            outputs=tuple(TensorFeatures(shape=t.shape, dtype=t.dtype, device=t.device) for t in outputs),
            attributes=(("selected_blocks", selected),),
        )
        parsed = adaptor.parse(features)
        estimate = model(
            _OVERHEAD_US,
            VecOpConfig(**_SLA2_VEC_CONFIG_KWARGS),
            semantics=parsed,
            hardware=hardware,
        ).predict()
        details = estimate.details
        if details is None:
            return CostEstimate.unknown_cost()
        flops = int((details.cube_flops or 0) + (details.vec_flops or 0))
        traffic = int((details.read_bytes or 0) + (details.write_bytes or 0))
        return CostEstimate(flops=flops, traffic_bytes=traffic, duration_us=estimate.duration_us)
    except Exception:
        return CostEstimate.unknown_cost()


def _sla2_sparse_attention_cost(
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware=None,
    attrs: dict | None = None,
) -> CostEstimate:
    """``fused.sla2_sparse_attention`` (+ grad) via the library model."""
    return _sla2_library_cost(
        "sla2_sparse_attention" if len(outputs) == 1 else "sla2_sparse_attention_gradient",
        inputs,
        outputs,
        hardware,
        attrs,
    )


def _sla2_linear_attention_cost(
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware=None,
    attrs: dict | None = None,
) -> CostEstimate:
    """``fused.sla2_linear_attention`` (+ grad) via the library model."""
    return _sla2_library_cost(
        "sla2_linear_attention" if len(outputs) == 1 else "sla2_linear_attention_gradient",
        inputs,
        outputs,
        hardware,
        attrs,
    )


# Handlers take (inputs, outputs, hardware) and optionally attrs; runtime
# dispatch checks the signature, so the wide Callable keeps both arities.
_HANDLERS: dict[str, Callable[..., CostEstimate]] = {
    # Dispatch nodes carry the normalized ``op_type`` ("matmul") alongside the
    # raw ``aten.mm.default`` annotation; the registry keys on the former.
    "matmul": _matmul_cost,
    "bmm": _bmm_cost,
    "grouped_mm": _grouped_mm_cost,
    # Declared-family dispatch spellings: routing only, the per-op weights
    # live in the package's declared tables.
    **{key: _declared_cost(canonical, raw) for key, (canonical, raw) in _DECLARED_DISPATCH.items()},
    # SLA2 branch fusions (multi_head_moe): modeled entirely by the
    # operator-performance-models package (fwd/grad families); needs chip
    # capabilities, so nodes stay cost_unknown without a hardware profile.
    "sla2_sparse_attention": _sla2_sparse_attention_cost,
    "sla2_linear_attention": _sla2_linear_attention_cost,
}


def _normalize_raw(raw: str) -> str:
    name = raw
    for prefix in ("aten.", "npu.", "cann_ops_transformer.", "torchtitan."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.removesuffix(".default")


_DECLARED_INDEX: dict[str, str] | None = None


def _declared_index() -> dict[str, str]:
    """Raw spelling -> declared fallback canonical (package-owned table)."""
    global _DECLARED_INDEX
    if _DECLARED_INDEX is None:
        from operator_performance_models.adaptors.vec.declared import declared_raw_index

        _DECLARED_INDEX = declared_raw_index()
    return _DECLARED_INDEX


def _raw_cost(
    raw: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware,
    attrs: dict | None = None,
) -> CostEstimate | None:
    """Route a raw spelling through the library: precise family first, then
    the declared fallback.

    The builtin raw index resolves the spelling to the precise structural
    family where one exists (matmul, convolution, elementwise, reductions,
    GDR, ...).  A precise family that requires captured scalars this route
    does not carry rejects the node, and the declared family for the same
    spelling (the package's documented fallback) takes over; ops nothing
    claims stay unknown.  The spelling passed to the library is the one that
    matched, so each family's own raw-name contract stays intact.
    """
    index = _raw_index()
    canonical = index.get(raw)
    spelling = raw
    if canonical is None:
        name = _normalize_raw(raw)
        if name != raw:
            canonical = index.get(name)
            spelling = name
    if canonical is not None:
        estimate = _routed_cost(canonical, spelling, inputs, outputs, hardware, attrs)
        if estimate is not None:
            return estimate
    declared = _declared_index()
    declared_canonical = declared.get(spelling) or declared.get(raw)
    if declared_canonical is None or declared_canonical == canonical:
        return None
    return _routed_cost(declared_canonical, spelling, inputs, outputs, hardware, attrs)


def _accepts_attrs(handler: Callable) -> bool:
    """SLA2-style handlers take the captured node attributes as a 4th argument."""
    try:
        return len(signature(handler).parameters) == 4
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False


def compute_cost(
    op_type: str,
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware=None,
    raw_op_type: str | None = None,
    attrs: dict | None = None,
) -> CostEstimate:
    """Look up the handler for one dispatch op; unknown ops stay unknown.

    Passing a hardware profile additionally asks the library for a duration
    prediction; the default ``None`` keeps every call hardware-independent.
    A dispatch handler that rejects the node (or returns unknown) falls back
    to the captured raw spelling route before the op stays unknown, so a
    stale declared dispatch cannot mask a precise family the package knows.
    Handlers that declare an ``attrs`` parameter receive the captured node
    attributes (schema scalars of virtual fusions).
    """
    estimate: CostEstimate | None = None
    handler = _HANDLERS.get(op_type)
    if handler is not None:
        try:
            if _accepts_attrs(handler):
                estimate = handler(inputs, outputs, hardware, attrs or {})
            else:
                estimate = handler(inputs, outputs, hardware)
        except (IndexError, TypeError, ValueError):
            estimate = None
    if estimate is None or estimate.unknown or (
        not estimate.flops and not estimate.traffic_bytes and estimate.duration_us is None
    ):
        raw = raw_op_type or op_type
        try:
            raw_estimate = _raw_cost(raw, inputs, outputs, hardware, attrs)
            if raw_estimate is None and raw_op_type is not None and raw_op_type != op_type:
                raw_estimate = _raw_cost(op_type, inputs, outputs, hardware, attrs)
            if raw_estimate is not None:
                return raw_estimate
        except (IndexError, TypeError, ValueError):
            if estimate is None:
                return CostEstimate.unknown_cost()
    if estimate is None:
        return CostEstimate.unknown_cost()
    return estimate


def _annotate_node(node: OpNode, hardware) -> None:
    cost = compute_cost(
        node.op_type,
        node.inputs,
        node.outputs,
        hardware,
        node.annotations.get("raw_op_type"),
        attrs=node.attrs,
    )
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

    Explicit and idempotent: ops without a route get ``cost_unknown`` only.
    Passing ``hardware`` (see :mod:`torchtitan_simulator.cost.hardware`) also
    stamps ``duration_us`` on ops the library supports.
    """
    for graph in workload.step_templates.values():
        for node in graph.nodes.values():
            _annotate_node(node, hardware)
    for graph in workload.captured_workload.step_templates.values():
        for node in graph.nodes.values():
            _annotate_node(node, hardware)
