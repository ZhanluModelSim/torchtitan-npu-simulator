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
    internal_data_passes: list = field(default_factory=list)  # CP comm DataPasses within this StepGraph

    def __post_init__(self) -> None:
        if self.nodes and (not self.entry_nodes or not self.exit_nodes):
            self.entry_nodes, self.exit_nodes = _compute_entry_exit(self.nodes)
        if self.nodes:
            self.is_acyclic = _check_acyclic(self.nodes)

    def export_l0_csv(self, path: str) -> None:
        """Export L1→L0: all L0 OpNodes in this step, in topological order.

        Columns: topo_order, op_id, seq_idx, op_type, raw_op_type, phase,
        inputs_shape, outputs_shape, inputs_dtype, outputs_dtype,
        flops, peak_mem, comm_bytes, repeat_count, module_path,
        comm_dim, comm_ranks, predecessors, successors
        """
        import csv
        from collections import deque

        # Topological sort (Kahn's algorithm, ties by op_id ascending)
        in_degree = {op_id: sum(1 for p in node.predecessors if p in self.nodes) for op_id, node in self.nodes.items()}
        ready = sorted(op_id for op_id, deg in in_degree.items() if deg == 0)
        queue: deque[int] = deque(ready)
        sorted_ids: list[int] = []
        while queue:
            op_id = queue.popleft()
            sorted_ids.append(op_id)
            newly_ready = []
            for succ in self.nodes[op_id].successors:
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        newly_ready.append(succ)
            if newly_ready:
                queue = deque(sorted(list(queue) + sorted(newly_ready)))

        shapes = lambda metas: ";".join("[" + ",".join(str(d) for d in m.shape) + "]" for m in metas)
        dtypes = lambda metas: ";".join(m.dtype for m in metas)

        # When fused_regions is populated, append region_id/fused_op_type columns
        # so each op row shows which fusion region it belongs to (additive; the
        # base columns stay identical to the non-fusion export).
        fused = bool(getattr(self, "fused_regions", None))
        op_to_region: dict[int, tuple[int, str]] = {}
        if fused:
            for r in self.fused_regions:
                for oid in r.op_ids:
                    op_to_region[oid] = (r.region_id, r.fused_op_type)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "topo_order", "op_id", "seq_idx", "op_type", "raw_op_type", "phase",
                "comp_type", "fsdp_state",
                "inputs_shape", "outputs_shape", "inputs_dtype", "outputs_dtype",
                "flops", "peak_mem", "comm_bytes", "repeat_count",
                "module_path", "comm_dim", "comm_ranks",
                "predecessors", "successors",
            ]
            if fused:
                header += ["region_id", "fused_op_type"]
            w.writerow(header)
            for topo_idx, op_id in enumerate(sorted_ids):
                node = self.nodes[op_id]
                ann = node.annotations
                row = [
                    topo_idx, op_id, node.seq_idx, node.op_type,
                    ann.get("raw_op_type", ""), ann.get("phase", ""),
                    ann.get("comp_type", ""), ann.get("fsdp_state", "NA"),
                    shapes(node.inputs), shapes(node.outputs),
                    dtypes(node.inputs), dtypes(node.outputs),
                    node.flops, node.peak_mem, node.comm_bytes,
                    ann.get("repeat_count", 1), ann.get("module_path", ""),
                    ann.get("comm_dim", ""), ann.get("comm_ranks", ""),
                    ";".join(str(p) for p in node.predecessors),
                    ";".join(str(s) for s in node.successors),
                ]
                if fused:
                    rid, ftype = op_to_region.get(op_id, ("", ""))
                    row += [rid, ftype]
                w.writerow(row)

    def export_fused_regions_csv(self, path: str) -> None:
        """Export the fusion regions: one row per fused region. Mirrors the L0
        export granularity (one CSV per step) so the fused result is
        inspectable in the same format as step_*_l0_ops.csv.

        Columns: region_id, fused_op_type, is_unfused, op_count, op_ids,
        eliminated_intermediates_bytes, member_op_types
        """
        import csv

        regions = getattr(self, "fused_regions", None) or []
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "region_id", "fused_op_type", "is_unfused", "op_count", "op_ids",
                "eliminated_intermediates_bytes", "member_op_types",
            ])
            for r in regions:
                types = {}
                for oid in r.op_ids:
                    t = self.nodes[oid].op_type if oid in self.nodes else "?"
                    types[t] = types.get(t, 0) + 1
                w.writerow([
                    r.region_id, r.fused_op_type, int(r.is_unfused),
                    len(r.op_ids), ";".join(str(o) for o in r.op_ids),
                    r.eliminated_intermediates_bytes,
                    ";".join(f"{k}:{v}" for k, v in types.items()),
                ])
