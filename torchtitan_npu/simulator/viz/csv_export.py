# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-rank topological-order CSV export of all L0-L3 graph nodes.

Outputs ``kernel_summary.csv``: one row per (rank, step_template, op_node),
sorted by rank → step template order → topological order (ties broken by
op_id ascending).  This gives a flat, spreadsheet-friendly view of every
kernel every rank executes, in execution dependency order.

The topological order is computed per StepGraph via Kahn's algorithm
(restricted to in-graph edges, mirroring ``step_graph._check_acyclic``).
Nodes with no in-graph predecessors get topo_order 0; their successors
get 1; and so on.  Ties (multiple nodes at the same depth) are broken
by op_id ascending, ensuring deterministic output across runs.
"""

from __future__ import annotations

import csv
from collections import deque

from torchtitan_npu.simulator.capture.op_mapping import display_op_label
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph


def _topo_sort(nodes: dict) -> list[str]:
    """Return op_ids in topological order (Kahn's algorithm).  Ties broken
    by op_id ascending.  External predecessors (not in ``nodes``) are
    treated as already-satisfied, mirroring ``_check_acyclic``."""
    in_degree: dict[str, int] = {
        op_id: sum(1 for p in node.predecessors if p in nodes)
        for op_id, node in nodes.items()
    }
    # Use a sorted list as the ready-queue so ties break by op_id ascending
    ready = sorted(op_id for op_id, deg in in_degree.items() if deg == 0)
    queue: deque[str] = deque(ready)
    result: list[str] = []
    while queue:
        op_id = queue.popleft()
        result.append(op_id)
        # Collect successors whose in-degree drops to 0
        newly_ready: list[str] = []
        for succ in nodes[op_id].successors:
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    newly_ready.append(succ)
        # Insert in sorted order to maintain tie-break invariant
        if newly_ready:
            merged = sorted(newly_ready)
            # Merge into queue maintaining sorted order
            remaining = sorted(queue)
            queue = deque(merged + remaining)
            # Re-sort: deque doesn't guarantee order, so rebuild
            queue = deque(sorted(list(queue)))
    return result


def _shapes_str(metas: list) -> str:
    """Compact representation of tensor shapes: ``[1,1024];[128,7168]``."""
    return ";".join("[" + ",".join(str(d) for d in m.shape) + "]" for m in metas)


def export_kernel_summary_csv(workload_graph: WorkloadGraph, path: str, *, max_ranks: int | None = None) -> None:
    """Write ``kernel_summary.csv`` with one row per (rank, step, op).

    Columns:
        rank, step_type, step_id, topo_order, op_id, op_type, raw_op_type,
        inputs_shape, outputs_shape, inputs_dtype, outputs_dtype,
        flops, peak_mem, param_mem, comm_bytes, repeat_count, module_path, phase

    Args:
        workload_graph: The captured L3 WorkloadGraph.
        path: Output CSV file path.
        max_ranks: If set, only expand the first ``max_ranks`` ranks instead
            of all ``world_size`` ranks.  Useful for large-scale simulations
            (e.g. 2048 dies) where the full per-rank CSV would be enormous.
            When None, all ranks are expanded.
    """
    schedule = workload_graph.iteration.schedule
    rank_table = schedule.annotations.get("rank_table", {})
    world_size = rank_table.get("world_size", 1) if isinstance(rank_table, dict) else 1
    if max_ranks is not None:
        world_size = min(world_size, max_ranks)

    template_ids = list(workload_graph.step_templates.keys())

    # Pre-compute topo-sorted op_ids per template (same for all ranks since
    # all ranks share the same template)
    topo_orders: dict[str, list[str]] = {}
    for tid in template_ids:
        step_graph = workload_graph.step_templates[tid]
        topo_orders[tid] = _topo_sort(step_graph.nodes)

    # Pre-compute row data per template (rank-independent fields)
    template_rows: dict[str, list[list]] = {}
    for tid in template_ids:
        step_graph = workload_graph.step_templates[tid]
        nodes = step_graph.nodes
        sorted_ids = topo_orders[tid]
        rows = []
        for topo_idx, op_id in enumerate(sorted_ids):
            node = nodes[op_id]
            ann = node.annotations
            # Extract communication info from annotations (set by comm_events)
            comm_dim = ann.get("comm_dim", "")
            comm_ranks = ann.get("comm_ranks", "")
            rows.append([
                step_graph.step_type,
                tid,
                topo_idx,
                op_id,
                display_op_label(node.op_type, ann),
                ann.get("raw_op_type", ""),
                _shapes_str(node.inputs),
                _shapes_str(node.outputs),
                ";".join(m.dtype for m in node.inputs),
                ";".join(m.dtype for m in node.outputs),
                node.flops,
                node.peak_mem,
                node.param_mem,
                node.comm_bytes,
                ann.get("repeat_count", 1),
                ann.get("module_path", ""),
                ann.get("phase", ""),
                comm_dim,
                comm_ranks,
            ])
        template_rows[tid] = rows

    # Write with large buffered I/O to avoid per-row syscall overhead
    # (critical for large world_size × node_count)
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "rank", "step_type", "step_id", "topo_order", "op_id", "op_type",
        "raw_op_type", "inputs_shape", "outputs_shape",
        "inputs_dtype", "outputs_dtype",
        "flops", "peak_mem", "param_mem", "comm_bytes",
        "repeat_count", "module_path", "phase",
        "comm_dim", "comm_ranks",
    ])

    BATCH_SIZE = 10000
    rows_in_buffer = 0

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(buf.getvalue())
        buf = io.StringIO()
        writer = csv.writer(buf)

        for rank in range(world_size):
            for tid in template_ids:
                for row in template_rows[tid]:
                    writer.writerow([rank] + row)
                    rows_in_buffer += 1
                    if rows_in_buffer >= BATCH_SIZE:
                        f.write(buf.getvalue())
                        buf = io.StringIO()
                        writer = csv.writer(buf)
                        rows_in_buffer = 0

        if rows_in_buffer > 0:
            f.write(buf.getvalue())
