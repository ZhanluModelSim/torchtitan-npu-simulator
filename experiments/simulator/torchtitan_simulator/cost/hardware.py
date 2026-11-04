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
- HBM 3.07 TB/s with a constant 0.9 efficiency -- datasheet assumption, NOT
  calibrated (no bandwidth-bound calibration point exists yet).
- vec peak is an uncalibrated placeholder: no vec duration model is wired in
  this module, the value only satisfies the ChipCapabilities structure.
- ``overhead_us`` 3.0 -- the kernel-duration floor observed in NPU traces.

Honest measured-vs-predicted gaps (model/real, >1 means the model is longer):

- qwen3.5 dense trace (372 mm): aggregate 0.81x; per-kernel median 1.22x,
  p10-p90 0.71-2.08x.
- deepseek_v4.1 ep4 trace (1696 mm, held-out model): aggregate 0.92x;
  median 1.00x, p10-p90 0.65-1.50x.
- Idle deep-K / tiny-M families (64,248320,256), (2,256,248320) etc.:
  0.18x-0.42x -- the model underestimates because the 1-D FLOPs curve
  cannot express shape structure.
- The >1 tail on small GEMMs is expected for a cache-free roofline: real
  back-to-back kernels reuse L2-resident operands, while the model charges
  cold logical traffic.
"""

from __future__ import annotations

__all__ = ["npu_910b3"]

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
            utilization=ConstantUtilization(ratio=0.9),
        ),
    )
    return Hardware(name="npu_910b3", chip=chip)


def npu_910b3():
    """The calibrated 910B3 profile for duration prediction."""
    return _profile()
