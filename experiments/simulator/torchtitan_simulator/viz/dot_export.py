# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Renders each L1 StepGraph template's device-kernel DAG as Graphviz DOT.
Nodes are colored by op_type category (compute=lightblue, communication
=gold, data-move/memory=plum), matching the color scheme convention already
used by comparable trace tooling in this ecosystem (see design doc §5.9)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan_simulator.capture.op_mapping import display_op_label

if TYPE_CHECKING:
    from torchtitan_simulator.ir.workload_graph import WorkloadGraph

_COMM_OP_TYPES = {"allreduce", "allgather", "reduce_scatter", "all_to_all"}
_DATA_MOVE_OP_TYPES = {
    "moe_token_permute",
    "moe_token_unpermute",
    "moe_re_routing",
    "view",
    "reshape",
    "transpose",
    "cat",
    "split",
}


def _node_color(op_type: str) -> str:
    if op_type in _COMM_OP_TYPES:
        return "gold"
    if op_type in _DATA_MOVE_OP_TYPES:
        return "plum"
    return "lightblue"


def _visible_successors(nodes, op_id: int, visible_ids: set[int]) -> set[int]:
    """Follow alias-only view nodes until reaching visible device work."""
    successors: set[int] = set()
    pending = list(nodes[op_id].successors)
    visited: set[int] = set()
    while pending:
        successor_id = pending.pop()
        if successor_id in visited or successor_id not in nodes:
            continue
        visited.add(successor_id)
        if successor_id in visible_ids:
            successors.add(successor_id)
        else:
            pending.extend(nodes[successor_id].successors)
    return successors


def export_dot(workload_graph: WorkloadGraph, path: str, *, include_metadata_views: bool = False) -> None:
    lines = ["digraph ComputeGraph {", '  rankdir="LR";']
    nodes = {
        op_id: node for template in workload_graph.step_templates.values() for op_id, node in template.nodes.items()
    }
    visible_ids = {
        op_id
        for op_id, node in nodes.items()
        if include_metadata_views or not node.annotations.get("metadata_view", False)
    }
    for step_id, step_graph in workload_graph.step_templates.items():
        lines.append(f'  subgraph "cluster_{step_id}" {{')
        lines.append(f'    label="{step_graph.step_type}";')
        for op_id, node in step_graph.nodes.items():
            if op_id not in visible_ids:
                continue
            label = display_op_label(node.op_type, node.annotations)
            if node.annotations.get("repeat_count", 1) > 1:
                label += f" (x{node.annotations['repeat_count']})"
            lines.append(f'    "{op_id}" [label="{label}", style=filled, fillcolor={_node_color(node.op_type)}];')
        lines.append("  }")
    for op_id in sorted(visible_ids, key=str):
        successors = (
            nodes[op_id].successors if include_metadata_views else _visible_successors(nodes, op_id, visible_ids)
        )
        for successor in sorted(successors, key=str):
            if successor in visible_ids:
                lines.append(f'  "{op_id}" -> "{successor}";')
    lines.append("}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
