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
from inspect import signature

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
        estimate = MatmulRoofline(_OVERHEAD_US, semantics=parsed, hardware=hardware).predict()
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


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# GDN (gated delta rule) is a latency-bound sequential-scan kernel: state
# updates form a loop-carried dependency chain over tokens, so throughput
# rooflines do not apply.  The per-token latency constants come from the
# production trace (qwen3.5 dense, two independent kernel populations:
# 18 forward launches and 6 backward launches at T=512); idle-device
# microbenchmarks are unusable because DVFS drops the AIV clock ~3x on an
# idle part.  The kernel fixes d=dv=64/128 and value-block 16, so per-token
# latency is independent of B/H; only T and the AIV wave count matter.
_GDN_FWD_US_PER_TOKEN = 0.725
_GDN_BWD_US_PER_TOKEN = 3.224
_GDN_VALUE_BLOCK = 16
_GDN_AIV_CORES = 20


def _sla2_families() -> dict:
    """Import-guarded adaptor/model table for the four SLA2 library families.

    The structural formulas (cube GEMM populations, declared softmax/normalize
    vector units) live in ``operator-performance-models`` (see its
    ``docs/sla2-roofline-models.md``); this adapter only translates captured
    tensor facts and the ``selected_blocks`` schema scalar.
    """
    from operator_performance_models.adaptors.npu_fused import (
        Sla2LinearAttentionAdaptor,
        Sla2SparseAttentionAdaptor,
    )
    from operator_performance_models.adaptors.npu_gradients import (
        Sla2LinearAttentionGradientAdaptor,
        Sla2SparseAttentionGradientAdaptor,
    )
    from operator_performance_models.models.roofline.fused import (
        Sla2LinearAttentionGradientRoofline,
        Sla2LinearAttentionRoofline,
        Sla2SparseAttentionGradientRoofline,
        Sla2SparseAttentionRoofline,
    )

    return {
        "sla2_sparse_attention": (
            "Sla2SparseAttention",
            "multi_head_moe_sim.sla2_sparse_attention.default",
            Sla2SparseAttentionAdaptor(),
            Sla2SparseAttentionRoofline,
        ),
        "sla2_sparse_attention_gradient": (
            "Sla2SparseAttentionBackward",
            "multi_head_moe_sim.sla2_sparse_attention_grad.default",
            Sla2SparseAttentionGradientAdaptor(),
            Sla2SparseAttentionGradientRoofline,
        ),
        "sla2_linear_attention": (
            "Sla2LinearAttention",
            "multi_head_moe_sim.sla2_linear_attention.default",
            Sla2LinearAttentionAdaptor(),
            Sla2LinearAttentionRoofline,
        ),
        "sla2_linear_attention_gradient": (
            "Sla2LinearAttentionBackward",
            "multi_head_moe_sim.sla2_linear_attention_grad.default",
            Sla2LinearAttentionGradientAdaptor(),
            Sla2LinearAttentionGradientRoofline,
        ),
    }


# Declared vector configuration for the SLA2 fusions: equivalent_ops 1.0 means
# the library's structural unit counts are taken literally, and FP32 compute
# matches the debug model's FP32 parameters. Explicitly declared, not
# calibrated; a consumer may override by extending the cost layer.
_SLA2_VEC_CONFIG_KWARGS = {"compute_bits": 32, "equivalent_ops": 1.0, "compute_dtype": "float32"}

# Captured TensorMeta shorthand -> the library's canonical dtype names; the
# shared npu adaptor helpers only accept the canonical spellings.
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "f16": "float16",
    "fp32": "float32",
    "f32": "float32",
    "fp64": "float64",
    "f64": "float64",
}


def _canonical_dtype(dtype: str) -> str:
    return _DTYPE_ALIASES.get(dtype, dtype)


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
            inputs=tuple(
                TensorFeatures(shape=t.shape, dtype=_canonical_dtype(t.dtype), device=t.device) for t in inputs
            ),
            outputs=tuple(
                TensorFeatures(shape=t.shape, dtype=_canonical_dtype(t.dtype), device=t.device) for t in outputs
            ),
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
        flops = (details.cube_flops or 0) + (details.vec_flops or 0)
        traffic = (details.read_bytes or 0) + (details.write_bytes or 0)
        return CostEstimate(flops=flops, traffic_bytes=traffic, duration_us=estimate.duration_us)
    except Exception:
        return CostEstimate.unknown_cost()


def _sla2_sparse_attention_cost(
    inputs: list[TensorMeta],
    outputs: list[TensorMeta],
    hardware=None,
    attrs: dict | None = None,
) -> CostEstimate:
    """``multi_head_moe_sim.sla2_sparse_attention`` (+ grad) via the library model."""
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
    """``multi_head_moe_sim.sla2_linear_attention`` (+ grad) via the library model."""
    return _sla2_library_cost(
        "sla2_linear_attention" if len(outputs) == 1 else "sla2_linear_attention_gradient",
        inputs,
        outputs,
        hardware,
        attrs,
    )


def _gdn_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """``qwen3_5_sim.gated_delta_rule``: forward (1 output) and backward (5).

    FLOPs follow the kernel source (sequential state recursion: ~9 MAC-passes
    over d*dv per token per head for the forward; the backward re-runs the
    forward with checkpointing plus a reverse pass, ~5x).  Duration is the
    declared latency model, not a roofline leg.
    """
    if not inputs or not outputs:
        return CostEstimate.unknown_cost()
    q, v = inputs[0], inputs[2] if len(inputs) > 2 else None
    if q is None or v is None or len(q.shape) != 4 or len(v.shape) != 4:
        return CostEstimate.unknown_cost()
    if q.shape[-1] not in (64, 128) or q.shape[-1] != v.shape[-1]:
        return CostEstimate.unknown_cost()
    batch, tokens = q.shape[0], q.shape[1]
    heads, dim, value_dim = v.shape[2], q.shape[-1], v.shape[-1]
    programs = batch * heads * _ceil_div(value_dim, _GDN_VALUE_BLOCK)
    waves = max(1, _ceil_div(programs, _GDN_AIV_CORES))
    backward = len(outputs) == 5
    if not backward and len(outputs) != 1:
        return CostEstimate.unknown_cost()
    flops = 9 * batch * heads * tokens * dim * value_dim
    if backward:
        flops *= 5
    traffic = _traffic(list(inputs) + list(outputs))
    if hardware is None or hardware.chip is None:
        return CostEstimate(flops=flops, traffic_bytes=traffic)
    per_token = _GDN_BWD_US_PER_TOKEN if backward else _GDN_FWD_US_PER_TOKEN
    duration = per_token * tokens * waves
    return CostEstimate(flops=flops, traffic_bytes=traffic, duration_us=duration)


def _make_mhc_handler(backward: bool):
    def handler(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
        if len(inputs) < 4 or not outputs:
            return CostEstimate.unknown_cost()
        x, h_res = inputs[0], inputs[1]
        if len(x.shape) < 2 or len(h_res.shape) < 2:
            return CostEstimate.unknown_cost()
        n, d = h_res.shape[-1], x.shape[-1]
        if n <= 0 or d <= 0 or h_res.shape[-2] != n:
            return CostEstimate.unknown_cost()
        batch = 1
        for size in x.shape[:-1]:
            batch *= size
        equiv_ops = 2.0 * (n + 1) * (2.0 if backward else 1.0)
        traffic = _traffic(list(inputs) + list(outputs))
        elements = _elements(outputs)
        if not traffic or not elements:
            return CostEstimate.unknown_cost()
        return CostEstimate(
            flops=0,
            traffic_bytes=traffic,
            duration_us=_vec_duration(hardware, elements, equiv_ops, traffic),
        )

    return handler


# Handlers take (inputs, outputs, hardware) and optionally attrs; runtime
# dispatch checks the signature, so the wide Callable keeps both arities.
def _communication_cost(inputs: list[TensorMeta], outputs: list[TensorMeta], hardware=None) -> CostEstimate:
    """``all_to_all`` / ``reduce_scatter``: logical cross-rank volume, no link model.

    Each logical input leaves the rank once and each output arrives once, so
    the collectives' bytes are in+out at dtype widths.  Duration needs a
    communication link model (``CollectiveRooflineConfig`` + topology), which
    the current workloads do not consume; nodes carry bytes without a duration.
    """
    return CostEstimate(comm_bytes=_traffic([*inputs, *outputs]))


_HANDLERS: dict[str, Callable[..., CostEstimate]] = {
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
    # Latency-bound sequential-scan kernel: duration from the declared
    # per-token latency model (see _gdn_cost), not a roofline leg.
    "gated_delta_rule": _gdn_cost,
    # SLA2 branch fusions (multi_head_moe, opt-in via enable_sla2_fusion):
    # modeled entirely by the operator-performance-models package (fwd/grad
    # families); needs chip capabilities, so nodes stay cost_unknown without
    # a hardware profile. Inert in the default eager mode (no such nodes).
    "sla2_sparse_attention": _sla2_sparse_attention_cost,
    "sla2_linear_attention": _sla2_linear_attention_cost,
    # Collectives: logical cross-rank volume only; duration awaits a link model.
    "all_to_all": _communication_cost,
    "reduce_scatter": _communication_cost,
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

_MHC_RAW = {
    "mhc_post": _make_mhc_handler(backward=False),
    "mhc_post_backward": _make_mhc_handler(backward=True),
}


def _normalize_raw(raw: str) -> str:
    name = raw
    for prefix in ("aten.", "npu.", "cann_ops_transformer."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.removesuffix(".default")


def _raw_cost(raw: str, inputs: list[TensorMeta], outputs: list[TensorMeta], hardware) -> CostEstimate | None:
    name = _normalize_raw(raw)
    if name in _MHC_RAW:
        return _MHC_RAW[name](inputs, outputs, hardware)
    equiv = _RAW_EQUIV.get(name)
    if equiv is not None:
        return _pointwise_cost(equiv)(inputs, outputs, hardware)
    if name in _RAW_COPY:
        return _copy_cost()(inputs, outputs, hardware)
    if name in _RAW_WRITE_ONLY:
        return _copy_cost(write_only=True)(inputs, outputs, hardware)
    return None


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

    Passing a hardware profile additionally asks handlers for a duration
    prediction; the default ``None`` keeps every call hardware-independent.
    Nodes dispatching as ``unknown`` fall back to their ``raw_op_type``.
    Handlers that declare an ``attrs`` parameter receive the captured node
    attributes (schema scalars of virtual fusions).
    """
    handler = _HANDLERS.get(op_type)
    if handler is not None:
        try:
            if _accepts_attrs(handler):
                return handler(inputs, outputs, hardware, attrs or {})
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
    cost = compute_cost(
        node.op_type,
        node.inputs,
        node.outputs,
        hardware,
        node.annotations.get("raw_op_type"),
        attrs=node.attrs,
    )
    if cost.unknown or (not cost.flops and not cost.traffic_bytes and not cost.comm_bytes and cost.duration_us is None):
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
