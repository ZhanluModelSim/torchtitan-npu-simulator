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


def _topo_sort(nodes: dict[int, OpNode]) -> list[int]:
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
    """Write per-rank CSV files under a ``kernel_summary/`` subdirectory.

    Each rank gets its own file ``rank_{N}.csv``, so users can open a
    single rank's data without filtering a giant mixed file.

    If ``path`` ends with ``.csv``, the ``.csv`` suffix is stripped and
    a directory is created at that base path.  For example, passing
    ``"out/kernel_summary.csv"`` creates ``out/kernel_summary/rank_0.csv``,
    ``out/kernel_summary/rank_1.csv``, etc.

    Columns:
        rank, step_type, step_id, topo_order, op_id, op_type, raw_op_type,
        inputs_shape, outputs_shape, inputs_dtype, outputs_dtype,
        flops, peak_mem, param_mem, comm_bytes, repeat_count, module_path,
        phase, comm_dim, comm_ranks

    Args:
        workload_graph: The captured L3 WorkloadGraph.
        path: Output base path (a ``kernel_summary/`` directory is created).
        max_ranks: If set, only expand the first ``max_ranks`` ranks.
    """
    import io
    import os

    schedule = workload_graph.iteration.schedule
    rank_table = schedule.annotations.get("rank_table", {})
    world_size = rank_table.get("world_size", 1) if isinstance(rank_table, dict) else 1
    if max_ranks is not None:
        world_size = min(world_size, max_ranks)

    template_ids = list(workload_graph.step_templates.keys())

    # Pre-compute topo-sorted op_ids per template (same for all ranks)
    topo_orders: dict[str, list[str]] = {}
    for tid in template_ids:
        step_graph = workload_graph.step_templates[tid]
        topo_orders[tid] = _topo_sort(step_graph.nodes)

    # Pre-compute row data per template (rank-independent fields).
    # Each row is a list of strings ready for csv.writer -- the rank
    # column is prepended per-file.
    HEADER = [
        "rank", "step_type", "step_id", "topo_order", "op_id", "op_type",
        "raw_op_type", "inputs_shape", "outputs_shape",
        "inputs_dtype", "outputs_dtype",
        "flops", "peak_mem", "param_mem", "comm_bytes",
        "repeat_count", "module_path", "phase",
        "comm_dim", "comm_ranks",
    ]

    template_rows: dict[str, list[list[str]]] = {}
    for tid in template_ids:
        step_graph = workload_graph.step_templates[tid]
        nodes = step_graph.nodes
        sorted_ids = topo_orders[tid]
        rows = []
        for topo_idx, op_id in enumerate(sorted_ids):
            node = nodes[op_id]
            ann = node.annotations
            rows.append([
                step_graph.step_type,
                tid,
                str(topo_idx),
                op_id,
                display_op_label(node.op_type, ann),
                ann.get("raw_op_type", ""),
                _shapes_str(node.inputs),
                _shapes_str(node.outputs),
                ";".join(m.dtype for m in node.inputs),
                ";".join(m.dtype for m in node.outputs),
                str(node.flops),
                str(node.peak_mem),
                str(node.param_mem),
                str(node.comm_bytes),
                str(ann.get("repeat_count", 1)),
                ann.get("module_path", ""),
                ann.get("phase", ""),
                ann.get("comm_dim", ""),
                ann.get("comm_ranks", ""),
            ])
        template_rows[tid] = rows

    # Determine output directory: strip ".csv" suffix if present, then
    # use the base path as a directory.
    base = path
    if base.endswith(".csv"):
        base = base[:-4]
    out_dir = base if os.path.splitext(base)[1] == "" else base + "_dir"
    os.makedirs(out_dir, exist_ok=True)

    # Write one file per rank.  Since all ranks share the same template
    # rows (only the rank column differs), we pre-serialize the template
    # rows into a single string and prepend the rank column per file --
    # this avoids re-iterating the Python row list for every rank.
    # Pre-serialize: build a list of (prefix, row_str) tuples where
    # prefix is "{rank}," and row_str is the CSV-serialized row without
    # the rank column.
    import csv as _csv

    # Serialize template rows once (without rank column)
    serialized_templates: list[str] = []  # one CSV line per op, across all templates
    buf = io.StringIO()
    writer = _csv.writer(buf)
    for tid in template_ids:
        for row in template_rows[tid]:
            writer.writerow(row)
    # Split into individual lines (each line is one op's row without rank)
    all_lines = buf.getvalue().splitlines(keepends=True)

    header_line = ",".join(HEADER) + "\n"

    for rank in range(world_size):
        rank_file = os.path.join(out_dir, f"rank_{rank}.csv")
        with open(rank_file, "w", encoding="utf-8") as f:
            f.write(header_line)
            # Write all lines with rank prefix
            rank_prefix = f"{rank},"
            # Batch write for efficiency
            BATCH = 8192
            for i in range(0, len(all_lines), BATCH):
                chunk = "".join(rank_prefix + line for line in all_lines[i : i + BATCH])
                f.write(chunk)
