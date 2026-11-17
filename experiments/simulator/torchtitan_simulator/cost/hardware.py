# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hardware profiles for the *explicit* duration-prediction path.

The simulator itself models no timing; :mod:`torchtitan_simulator.cost`
annotates workload facts, and duration prediction is an opt-in layer that
consumes those facts through the workspace's ``operator-performance-models``
roofline package (``max(cube, memory) + overhead`` with log-utilization
curves).  The profile below is calibrated against idle-NPU GEMM measurements
taken on the real device (qwen35-real container, BF16):

- cube BF16 peak 294.912 TFLOPS -- the 910B3 figure used across this workspace
  (torchtitan-npu ``metrics.py``); the largest measured GEMM (8192^3) sustains
  260 TFLOPS, so the peak is not overreached.
- cube utilization ``LogUtilization(k, b, min_ratio, max_ratio)`` -- calibrated
  exclusively on idle-device BF16 GEMM microbenchmarks (qwen35-real container,
  back-to-back launch timing): a square ladder 256..8192 plus a 3x3
  MxN grid at K=4096, cube-bound points only (n=16).  The fit set is chosen
  before evaluation and contains no target-workload shapes; residuals on it
  are median 0.92x, p90 3.54x (clamp region).  The 1-D FLOPs law is a known
  simplification: same-FLOPs shape families deviate from it (deep-K / tiny-M
  GEMMs run 3-6x below the curve, quantified below).
- HBM 3.07 TB/s peak with a constant 0.40 efficiency -- calibrated on
  idle-device contiguous bf16 copy (read+write) at 256MB/1GB: the stream
  sustains 1.22-1.24 TB/s.  Sizes up to ~64MB run faster (up to 4.3 TB/s,
  L2-resident operands); that cache boost is intentionally not modeled, so
  small-kernel copy time is overestimated.
- vec peak is an uncalibrated placeholder: every vec-family model in this
  module is memory-bound at the declared equivalent-op counts (verified per
  family), so the placeholder never dominates; it only satisfies the
  ChipCapabilities structure.
- ``overhead_us`` 3.0 -- the kernel-duration floor observed in NPU traces.

Honest measured-vs-predicted gaps (model/real, >1 means the model is longer),
validated per trace kernel on the qwen3.5 dense trace (66k mapped kernels):

- matmul: aggregate 0.75x (5.25 vs 7.02 ms); per-kernel median 1.22x,
  p10-p90 0.70-2.19x.
- adamw: per-kernel median 0.85x -- but the well-streaming fused kernel
  sustains 2.3 TB/s, above the clone-calibrated copy bandwidth, so the
  sim-side fused step (whole model, one node) lands ~2.1x over the measured
  optimizer time.
- fused vec activations (silu family, rsqrt, exp): 1.8-2.4x over -- the
  real fused kernels run 0.9-1.7 us, below the flat 3 us launch overhead.
- reductions (sum/mean) and log-softmax: 0.24-0.50x (2-4x under) -- multi-pass
  kernels that do not reach copy bandwidth with single-pass traffic.
- copy/transform kernels (TransData/Cast/Slice/Transpose...): ~0.45x
  (2.2x under) -- layout conversion runs well below copy bandwidth and the
  real per-kernel floor is 5-8 us, not 3 us.
- Known capture-side gaps the cost model cannot see: MemSet (167 ms/step,
  shape-less in the trace), PadV3 (63 ms), Conv2D family (36 ms) and other
  unmodeled kernel types (107 ms total) have no corresponding workload nodes
  or no clean semantics yet.

Two fused families step outside the roofline path:

- gated_delta_rule (GDN): a latency-bound sequential-scan kernel (loop-carried
  state dependency), modeled as ``per_token_us * T * AIV_waves`` with
  constants from the production trace (forward 0.725 us/token, backward
  3.224 us/token, two independent launch populations at T=512).  Idle-device
  microbenchmarks are unusable here: DVFS drops the AIV clock ~3x on an idle
  part, so the same kernel measures 1.1 ms instead of 0.37 ms.  The linear-in-T
  shape follows from the kernel source; only the T=512 point is trace-validated.
- mhc_post (CANN fused head composition, deepseek_v4_1 only): closed-form
  equivalent-op model derived from the CANN kernel source
  (``x_{l+1}[i,:] = sum_j H_res[j,i] * x_l[j,:] + h_out * H_post[i]``);
  no qwen3.5 presence, so it carries unit tests but no trace validation.
"""

from __future__ import annotations

__all__ = ["dv120_hardware", "npu_910b3"]

_CUBE_LOG_UTILIZATION = (0.1056, -1.8626, 0.0404, 0.9254)


def _profile():
    try:
        from operator_performance_models.core import Hardware
        from operator_performance_models.hardware import (
            ChipCapabilities,
            ComputeCapability,
            ConstantUtilization,
            HbmCapability,
            LogUtilization,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "Duration prediction needs the operator-performance-models package"
            " (editable-installed in the workspace containers)"
        ) from exc

    k, b, min_ratio, max_ratio = _CUBE_LOG_UTILIZATION
    chip = ChipCapabilities(
        cube=ComputeCapability(
            tflops_by_bits=((16, 294.912),),
            utilization=LogUtilization(k=k, b=b, min_ratio=min_ratio, max_ratio=max_ratio),
        ),
        # Uncalibrated placeholder; inert until a vec duration model consumes it.
        vec=ComputeCapability(
            tflops_by_bits=((16, 128.0),),
            utilization=ConstantUtilization(ratio=0.5),
        ),
        hbm=HbmCapability(
            capacity_gb=64.0,
            bandwidth_tb_per_second=3.07,
            utilization=ConstantUtilization(ratio=0.40),
        ),
    )
    return Hardware(name="npu_910b3", chip=chip)


def npu_910b3():
    """The calibrated 910B3 profile for duration prediction."""
    return _profile()


_DV120_CHIP = "/workspace/hardware/dv120-chip.txt"


def _dv120_utilization(spec: dict):
    """DV120 utilization record -> the library's immutable utilization model."""
    from operator_performance_models.hardware import ConstantUtilization, LogUtilization

    if "constant" in spec:
        return ConstantUtilization(ratio=float(spec["constant"]))
    return LogUtilization(
        k=float(spec["log_a"]),
        b=float(spec["log_b"]),
        min_ratio=float(spec["min_ratio"]),
        max_ratio=float(spec["max_ratio"]),
    )


def dv120_hardware(path=_DV120_CHIP):
    """Real DV120 profile parsed from the platform hardware file (no calibration added).

    Maps the DV120 JSON onto ``operator-performance-models`` chip capabilities:
    compute peaks in TFLOPS keyed by ``floatN`` storage width, per-unit
    log/constant utilization models, HBM size/bandwidth and the two head-tail
    constants.  The companion topology file and the L2 cache record are not
    consumed by the compute roofline path.
    """
    import json
    from pathlib import Path

    from operator_performance_models.core import Hardware
    from operator_performance_models.hardware import (
        ChipCapabilities,
        ComputeCapability,
        HbmCapability,
    )

    data = json.loads(Path(path).read_text())
    compute, memory, utilization = data["compute"], data["memory"], data["utilization"]

    def capability(section: str) -> ComputeCapability:
        entries = tuple(
            (int(bits.removeprefix("float")), float(value))
            for bits, value in sorted(
                compute[section].items(), key=lambda kv: int(kv[0].removeprefix("float"))
            )
        )
        return ComputeCapability(
            tflops_by_bits=entries, utilization=_dv120_utilization(utilization[section])
        )

    hbm = memory["hbm"]
    chip = ChipCapabilities(
        cube=capability("cube"),
        vec=capability("vector"),
        hbm=HbmCapability(
            capacity_gb=float(hbm["size"]),
            bandwidth_tb_per_second=float(hbm["bandwidth"]),
            utilization=_dv120_utilization(utilization["hbm"]),
        ),
        head_tail_time=float(data["head_tail_time"]),
        head_tail_time_for_fused_op=float(data["head_tail_time_for_fused_op"]),
    )
    return Hardware(name=Path(path).stem, chip=chip)
