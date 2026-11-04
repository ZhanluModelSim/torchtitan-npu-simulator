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
  on the actual workload distribution (372 measured ``MatMulV2`` kernels from
  the qwen3.5 dense trace, cube-dominated subset), NOT on idle-device
  microbenchmarks: a 1-D FLOPs curve fit to benchmark shapes missed the real
  trace population by an order of magnitude (chunked tiny-M GEMMs run ~10x
  more efficiently than same-FLOPs benchmark points).  On the fit set the
  model reproduces the measured aggregate to 0.98x, and on the held-out
  v4.1 ep4 trace to 0.97x aggregate (per-kernel median 0.9x, p90 1.6x).
  ``max_ratio`` is pinned to the workload's 98th percentile, so idle-device
  large GEMMs (which reach 89% utilization) are intentionally overestimated
  up to ~3.6x -- this profile predicts training traces, not synthetic
  microbenchmarks.
- HBM 3.07 TB/s with a constant 0.9 efficiency -- datasheet assumption, NOT
  calibrated (no bandwidth-bound calibration point exists yet).
- vec peak is an uncalibrated placeholder: no vec duration model is wired in
  this module, the value only satisfies the ChipCapabilities structure.
- ``overhead_us`` 3.0 -- the kernel-duration floor observed in NPU traces.
"""

from __future__ import annotations

__all__ = ["npu_910b3"]

_CUBE_LOG_UTILIZATION = (0.0620, -1.0760, 0.0434, 0.2461)


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
