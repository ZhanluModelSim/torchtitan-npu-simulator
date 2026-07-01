# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Renders every L1 StepGraph template's L0 operator DAG as Graphviz DOT.
Nodes are colored by op_type category (compute=lightblue, communication
=gold, data-move/memory=plum), matching the color scheme convention already
used by comparable trace tooling in this ecosystem (see design doc §5.9)."""

from __future__ import annotations

from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph

_COMM_OP_TYPES = {"allreduce", "allgather", "reduce_scatter", "all_to_all"}
_DATA_MOVE_OP_TYPES = {"moe_token_permute", "moe_token_unpermute", "moe_re_routing", "view", "reshape", "transpose", "cat", "split"}


def _node_color(op_type: str) -> str:
    if op_type in _COMM_OP_TYPES:
        return "gold"
    if op_type in _DATA_MOVE_OP_TYPES:
        return "plum"
    return "lightblue"


def export_dot(workload_graph: WorkloadGraph, path: str) -> None:
    lines = ["digraph ComputeGraph {", '  rankdir="LR";']
    for step_id, step_graph in workload_graph.step_templates.items():
        lines.append(f'  subgraph "cluster_{step_id}" {{')
        lines.append(f'    label="{step_graph.step_type}";')
        for op_id, node in step_graph.nodes.items():
            label = f"{node.op_type}"
            if node.annotations.get("repeat_count", 1) > 1:
                label += f" (x{node.annotations['repeat_count']})"
            lines.append(f'    "{op_id}" [label="{label}", style=filled, fillcolor={_node_color(node.op_type)}];')
        for op_id, node in step_graph.nodes.items():
            for succ in node.successors:
                lines.append(f'    "{op_id}" -> "{succ}";')
        lines.append("  }")
    lines.append("}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
