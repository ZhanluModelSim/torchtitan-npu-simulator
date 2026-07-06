# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L1 StepGraph: a bounded DAG for one forward/backward/optimizer step. See
spec/L1-StepGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchtitan_npu.simulator.ir.op_node import OpNode


def _compute_entry_exit(nodes: dict[int, OpNode]) -> tuple[list[int], list[int]]:
    """Entry nodes have no predecessors *within this graph*. A predecessor
    absent from `nodes` is external to this StepGraph -- e.g. a
    backward-phase op referencing a forward-phase activation, or an
    optimizer-phase op referencing a backward-phase gradient. Per
    spec/L1-StepGraph.md: "entry_node 的 input 无内部 producer：依赖链追溯
    到外部" -- external predecessors do not disqualify a node from being an
    entry point. Exit nodes have no successors (successors are only ever
    populated for in-graph nodes, so no such adjustment is needed there)."""
    entry = [op_id for op_id, node in nodes.items() if not any(p in nodes for p in node.predecessors)]
    exit_ = [op_id for op_id, node in nodes.items() if not node.successors]
    return entry, exit_


def _check_acyclic(nodes: dict[int, OpNode]) -> bool:
    """Kahn's algorithm restricted to in-graph edges: a predecessor that is
    not itself a key of `nodes` is external to this StepGraph and is
    treated as an already-satisfied prerequisite (not counted toward
    in-degree) -- otherwise every node with an external predecessor would
    never reach in-degree zero, and `_check_acyclic` would incorrectly
    report every backward/optimizer StepGraph as cyclic (this exact bug was
    caught by an end-to-end integration run during design: a real
    forward->backward->optimizer step produced `is_acyclic=False` for the
    backward and optimizer graphs before this fix, `True` after)."""
    in_degree = {op_id: sum(1 for p in node.predecessors if p in nodes) for op_id, node in nodes.items()}
    queue = [op_id for op_id, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        op_id = queue.pop(0)
        visited += 1
        for succ in nodes[op_id].successors:
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
    return visited == len(nodes)


@dataclass
class StepGraph:
    step_id: str
    step_type: str
    nodes: dict[int, OpNode]
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)
    tensor_lifetimes: dict[str, int] = field(default_factory=dict)
    total_flops: int = 0
    peak_active_mem: int = 0
    param_mem: int = 0
    comm_volume: int = 0
    device_placement: dict[str, int] = field(default_factory=dict)
    is_acyclic: bool = True
    annotations: dict[str, Any] = field(default_factory=dict)
    fused_regions: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.nodes and (not self.entry_nodes or not self.exit_nodes):
            self.entry_nodes, self.exit_nodes = _compute_entry_exit(self.nodes)
        if self.nodes:
            self.is_acyclic = _check_acyclic(self.nodes)
