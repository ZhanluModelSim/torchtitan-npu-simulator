# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregate full-duplex state transfers around the native optimizer update block."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.memory.optimizer_offload import apply_optimizer_offload_budget

if TYPE_CHECKING:
    from torchtitan_simulator.ir.workload_graph import WorkloadGraph


def insert_optimizer_transfers(workload: WorkloadGraph) -> WorkloadGraph:
    """Join load, update and writeback paths without changing native compute costs.

    The copies represent an ideal per-tensor pipeline, not a dependency claim
    that one tensor can be written back before it is loaded or updated.
    """
    annotations = workload.iteration.schedule.annotations
    if "optimizer_transfer_plan" in annotations:
        raise ValueError("Optimizer transfers have already been inserted")
    memory = apply_optimizer_offload_budget(annotations["memory_plan"])
    state_bytes = memory.optimizer_state_bytes
    assert state_bytes is not None
    source = {key: node for step in workload.step_templates.values() for key, node in step.nodes.items()}
    updates = {
        key
        for key, node in source.items()
        if node.annotations.get("optimizer_work_scope") == "actual_updated_parameters"
    }
    if not updates or any(source[key].annotations.get("comp_type") != "OPTIMIZER" for key in updates):
        raise ValueError("Optimizer offload requires identified native optimizer update nodes in O")
    owner = {key: step_id for step_id, step in workload.step_templates.items() for key in step.nodes}
    template_ids = {owner[key] for key in updates}
    if len(template_ids) != 1:
        raise ValueError("Optimizer offload requires a single optimizer update template")
    template_id = next(iter(template_ids))
    external_parents = {parent for key in updates for parent in source[key].predecessors if parent not in updates}
    # A shared gate cannot wait on a node downstream of an update. Such an
    # interleaved optimizer needs separate windows, rather than a silent cycle.
    pending = list(external_parents)
    visited: set[int] = set()
    while pending:
        key = pending.pop()
        if key in updates:
            raise ValueError("Optimizer offload does not support interleaved non-update preparation nodes")
        if key not in visited:
            visited.add(key)
            pending.extend(source[key].predecessors)
    nodes = {
        key: replace(
            node,
            predecessors=list(node.predecessors),
            successors=[],
            attrs=dict(node.attrs),
            annotations=dict(node.annotations),
        )
        for key, node in source.items()
    }
    captured = workload.captured_workload or workload
    raw_ids = [key for step in captured.step_templates.values() for key in step.nodes]
    first_id = max([*raw_ids, *nodes]) + 1
    stream_base = max((node.annotations.get("stream_id", 0) for node in nodes.values()), default=0) + 1
    modeled_ids: list[int] = []
    start_seq = min(source[key].seq_idx for key in updates)
    end_seq = max(source[key].seq_idx for key in updates)

    def add_node(name: str, parents: set[int], *, direction: str | None = None, exit_node: bool = False) -> int:
        key = first_id + len(modeled_ids)
        nodes[key] = OpNode(
            key,
            name,
            [],
            [],
            {"optimizer_transfer": {"direction": direction, "bytes": state_bytes}} if direction else {},
            sorted(parents),
            [],
            seq_idx=end_seq if exit_node else start_seq,
            annotations={
                "raw_op_type": f"simulator.{name}",
                "module_path": "",
                "phase": "optimizer",
                "execution_kind": "optimizer",
                "comp_type": "OPTIMIZER",
                "stream_id": stream_base + len(modeled_ids),
                "modeling_source": "optimizer_offload_full_duplex_v1",
            },
        )
        modeled_ids.append(key)
        return key

    if state_bytes:
        entry = add_node("stream_wait", external_parents)
        for key in updates:
            nodes[key].predecessors = sorted(set(nodes[key].predecessors) | {entry})
        load = add_node("optimizer_h2d", {entry}, direction="h2d")
        store = add_node("optimizer_d2h", {entry}, direction="d2h")
        exit_id = add_node("stream_wait", updates | {load, store}, exit_node=True)
        for key in updates:
            for successor in source[key].successors:
                if successor not in updates:
                    nodes[successor].predecessors = sorted(set(nodes[successor].predecessors) | {exit_id})
    for node in nodes.values():
        for parent in node.predecessors:
            nodes[parent].successors.append(node.op_id)
    summary = memory.to_summary_dict()
    steps = {
        key: replace(
            step,
            nodes={node_id: nodes[node_id] for node_id in (*step.nodes, *(modeled_ids if key == template_id else []))},
            entry_nodes=[],
            exit_nodes=[],
            annotations={**step.annotations, "memory_summary": summary},
        )
        for key, step in workload.step_templates.items()
    }
    updated = {
        **annotations,
        "memory_plan": memory,
        "memory_summary": summary,
        "optimizer_transfer_plan": {
            "policy": "full_duplex_aggregate_v1",
            "modeled_op_ids": modeled_ids,
            "update_op_ids": sorted(updates),
            "transfer_bytes_per_direction": state_bytes,
        },
    }
    if workload.schedule_plan is None:
        raise ValueError("Optimizer transfers require a SchedulePlan")
    logging.getLogger(__name__).info(
        "Optimizer offload: %d bytes per direction, %d update operators; aggregate full-duplex window, "
        "device state and transient buffer residency excluded",
        state_bytes,
        len(updates),
    )
    return replace(
        workload,
        step_templates=steps,
        schedule_plan=replace(workload.schedule_plan, step_templates=steps),
        iteration=replace(
            workload.iteration, schedule=replace(workload.iteration.schedule, step_templates=steps, annotations=updated)
        ),
    )
