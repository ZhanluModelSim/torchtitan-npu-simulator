# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L3 WorkloadGraph: the outermost container -- holds a ScheduleGraph
template plus iteration semantics and data-flow cadence. See
spec/L3-WorkloadGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph


@dataclass
class DataFlow:
    """Describes one input or output data stream of the workload."""

    source: str
    tensor_shape: tuple[int | str, ...]
    dtype: str
    volume_per_iter: int
    is_streaming: bool = False
    interleave_strategy: str = "synced"


@dataclass
class IterationSpec:
    """One training/inference iteration: which ScheduleGraph it runs, and
    how many microbatches it contains."""

    schedule: ScheduleGraph
    microbatch_count: int
    iteration_time_est: float = 0.0


@dataclass
class WorkloadGraph:
    """Top-level container for a complete workload: train/inference/rag/
    recommendation, iteration semantics, and cross-iteration data flow."""

    workload_id: str
    workload_type: str
    step_templates: dict[str, StepGraph]
    iteration: IterationSpec
    num_iterations: int
    warmup_iterations: int = 0
    data_inputs: list[DataFlow] = field(default_factory=list)
    data_outputs: list[DataFlow] = field(default_factory=list)
    cross_iter_passes: list[DataPass] = field(default_factory=list)
    total_runtime_est: float = 0.0
    total_cost_est: float = 0.0
