# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Human-readable plain-text summary of a captured WorkloadGraph: op
counts per step, FLOPs/memory/communication totals, and an explicit list
of "unrecognized" op types (never silently hidden -- see design doc §5.8
and §9's note about the sibling project's MockCostModel coverage gap)."""

from __future__ import annotations

from torchtitan_npu.simulator.capture.op_mapping import display_op_label
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph


def export_text_summary(workload_graph: WorkloadGraph) -> str:
    lines: list[str] = []
    lines.append(f"Workload: {workload_graph.workload_id} ({workload_graph.workload_type})")
    lines.append(f"Iterations: {workload_graph.num_iterations} (warmup={workload_graph.warmup_iterations})")
    lines.append("")

    unknown_op_types: set[str] = set()
    for step_id, step_graph in workload_graph.step_templates.items():
        total_flops = sum(node.flops for node in step_graph.nodes.values())
        total_comm_bytes = sum(node.comm_bytes for node in step_graph.nodes.values())
        total_op_output_bytes_estimate = sum(node.peak_mem for node in step_graph.nodes.values())
        lines.append(f"[{step_graph.step_type}] step={step_id} nodes={len(step_graph.nodes)}")
        lines.append(
            f"  total_flops={total_flops}  "
            f"total_op_output_bytes_estimate={total_op_output_bytes_estimate}  "
            f"total_comm_bytes={total_comm_bytes}"
        )
        if step_graph.peak_active_mem:
            lines.append(
                f"  active_bytes_peak={step_graph.peak_active_mem}  "
                f"persistent_param_bytes={step_graph.param_mem}"
            )
        lines.append(f"  is_acyclic={step_graph.is_acyclic}")
        for node in step_graph.nodes.values():
            if node.annotations.get("cost_unknown"):
                unknown_op_types.add(display_op_label(node.op_type, node.annotations))
        lines.append("")

    schedule = workload_graph.iteration.schedule
    lines.append(f"Schedule: {len(schedule.instances)} instances, {len(schedule.data_passes)} data passes")
    lines.append(
        f"  dp_degree={schedule.dp_degree} tp_degree={schedule.tp_degree} pp_degree={schedule.pp_degree} "
        f"pipeline_schedule={schedule.pipeline_schedule}"
    )
    comm_bytes_by_primitive: dict[str, int] = {}
    for data_pass in schedule.data_passes:
        volume = sum(slot.volume_bytes for slot in data_pass.slots)
        comm_bytes_by_primitive[data_pass.comm_primitive] = comm_bytes_by_primitive.get(data_pass.comm_primitive, 0) + volume
    for primitive, total_bytes in sorted(comm_bytes_by_primitive.items()):
        lines.append(f"  comm[{primitive}] total_bytes={total_bytes}")
    memory_summary = schedule.annotations.get("memory_summary", {})
    if memory_summary:
        lines.append(
            "  memory "
            f"active_bytes_peak={memory_summary.get('active_bytes_peak', 0)} "
            f"peak_seq_idx={memory_summary.get('peak_seq_idx', 0)} "
            f"persistent_param_bytes={memory_summary.get('persistent_param_bytes', 0)}"
        )
    lines.append("")

    if unknown_op_types:
        lines.append(f"Unrecognized op types ({len(unknown_op_types)}) -- cost not estimated for these:")
        for op_type in sorted(unknown_op_types):
            lines.append(f"  - {op_type}")
    else:
        lines.append("All captured op types were recognized by the cost model.")

    return "\n".join(lines) + "\n"


def write_text_summary(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_text_summary(workload_graph))
