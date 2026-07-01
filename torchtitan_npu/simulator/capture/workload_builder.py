# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wraps a captured L2 ScheduleGraph into the top-level L3 WorkloadGraph:
iteration semantics + dataloader-derived data flow. See design doc §5.7 and
spec/L3-WorkloadGraph.md."""

from __future__ import annotations

import uuid

from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import DataFlow, IterationSpec, WorkloadGraph


def build_workload_graph(
    *,
    schedule_graph: ScheduleGraph,
    step_templates: dict[str, StepGraph],
    local_batch_size: int,
    seq_len: int,
    num_micro_batches: int = 1,
) -> WorkloadGraph:
    """One captured training step, wrapped as a single-iteration
    `WorkloadGraph`. `num_iterations` is always 1 -- the simulator captures
    exactly one train step, per design doc §1."""
    input_flow = DataFlow(
        source="dataloader",
        tensor_shape=(local_batch_size, seq_len),
        dtype="int64",
        volume_per_iter=local_batch_size * seq_len * 8,  # int64 = 8 bytes/token
        is_streaming=True,
        interleave_strategy="synced",
    )
    output_flow = DataFlow(
        source="labels",
        tensor_shape=(local_batch_size, seq_len),
        dtype="int64",
        volume_per_iter=local_batch_size * seq_len * 8,
        is_streaming=True,
        interleave_strategy="synced",
    )

    iteration = IterationSpec(schedule=schedule_graph, microbatch_count=num_micro_batches)

    return WorkloadGraph(
        workload_id=uuid.uuid4().hex[:12],
        workload_type="train",
        step_templates=step_templates,
        iteration=iteration,
        num_iterations=1,
        warmup_iterations=0,
        data_inputs=[input_flow],
        data_outputs=[output_flow],
        cross_iter_passes=[],
    )
