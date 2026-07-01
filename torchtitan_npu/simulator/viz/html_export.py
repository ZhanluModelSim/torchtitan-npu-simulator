# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained HTML visualization of the captured four-layer IR: L3
workload card, L2 RankTable + schedule summary, L1 step cards, and a
foldable L0 operator listing per step (see design doc §5.9)."""

from __future__ import annotations

import html
import re

from torchtitan_npu.simulator.capture.op_mapping import display_op_label
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph

_NUMERIC_SEGMENT = re.compile(r"\.\d+(?=\.|$)")


def normalize_module_path(path: str) -> str:
    """Replace numeric `ModuleList` indices with `N` so repeated layers
    (e.g. `layers.0.attention`, `layers.1.attention`, ...) collapse to the
    same group key (`layers.N.attention`)."""
    if not path:
        return path
    return _NUMERIC_SEGMENT.sub(".N", path)


def _group_nodes_by_normalized_path(nodes: dict[str, OpNode]) -> dict[str, list[tuple[str, OpNode]]]:
    groups: dict[str, list[tuple[str, OpNode]]] = {}
    for op_id, node in nodes.items():
        raw_path = node.annotations.get("module_path", "")
        key = normalize_module_path(raw_path) or "(root)"
        groups.setdefault(key, []).append((op_id, node))
    return groups


def _distinct_raw_paths(entries: list[tuple[str, OpNode]]) -> set[str]:
    return {node.annotations.get("module_path", "") for _, node in entries}


def _render_op_row(op_id: str, node: OpNode) -> str:
    repeat = node.annotations.get("repeat_count", 1)
    repeat_suffix = f" (dedup x{repeat})" if repeat > 1 else ""
    unknown_suffix = " [cost unknown]" if node.annotations.get("cost_unknown") else ""
    label = display_op_label(node.op_type, node.annotations)
    return (
        f"<li><code>{html.escape(op_id)}</code> "
        f"<strong>{html.escape(label)}</strong>{repeat_suffix}{unknown_suffix} "
        f"flops={node.flops} peak_mem={node.peak_mem} comm_bytes={node.comm_bytes}</li>"
    )


def _render_step_graph_section(step_graph: StepGraph) -> str:
    groups = _group_nodes_by_normalized_path(step_graph.nodes)
    parts = [
        f"<h3>{html.escape(step_graph.step_type)} "
        f"(step_id={html.escape(step_graph.step_id)}, {len(step_graph.nodes)} ops, "
        f"is_acyclic={step_graph.is_acyclic})</h3>"
    ]
    for group_key in sorted(groups):
        entries = groups[group_key]
        distinct_paths = _distinct_raw_paths(entries)
        occurrence_count = max(len(distinct_paths), 1)
        summary_label = html.escape(group_key)
        if occurrence_count > 1:
            summary_label += f" &times; {occurrence_count} layers"
        # Render only the first occurrence's ops in full when the group is
        # a repeated layer; always render every op for non-repeated groups.
        if occurrence_count > 1:
            first_path = sorted(distinct_paths)[0]
            rows = [_render_op_row(op_id, node) for op_id, node in entries if node.annotations.get("module_path") == first_path]
        else:
            rows = [_render_op_row(op_id, node) for op_id, node in entries]
        parts.append(
            f"<details><summary>{summary_label} ({len(rows)} ops shown"
            f"{' for representative layer' if occurrence_count > 1 else ''})</summary>"
            f"<ul>{''.join(rows)}</ul></details>"
        )
    return "\n".join(parts)


def _render_rank_table_section(workload_graph: WorkloadGraph) -> str:
    schedule = workload_graph.iteration.schedule
    rank_table = schedule.annotations.get("rank_table", {}) if schedule.annotations else {}
    dim_degrees = rank_table.get("dim_degrees", {})
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(dim_degrees.items()))
    return (
        "<h2>L2: RankTable / Schedule</h2>"
        f"<p>world_size={rank_table.get('world_size', '?')}, "
        f"instances={len(schedule.instances)}, data_passes={len(schedule.data_passes)}, "
        f"dp_degree={schedule.dp_degree}, tp_degree={schedule.tp_degree}, pp_degree={schedule.pp_degree}, "
        f"pipeline_schedule={html.escape(schedule.pipeline_schedule)}</p>"
        f"<table border='1'><tr><th>dimension</th><th>degree</th></tr>{rows}</table>"
    )


def _render_workload_section(workload_graph: WorkloadGraph) -> str:
    inputs = "".join(
        f"<li>{html.escape(f.source)}: shape={f.tensor_shape} dtype={f.dtype} "
        f"volume_per_iter={f.volume_per_iter}</li>"
        for f in workload_graph.data_inputs
    )
    return (
        "<h2>L3: WorkloadGraph</h2>"
        f"<p>workload_id={html.escape(workload_graph.workload_id)} "
        f"type={html.escape(workload_graph.workload_type)} "
        f"num_iterations={workload_graph.num_iterations}</p>"
        f"<ul>{inputs}</ul>"
    )


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>torchtitan_npu simulator trace</title>
<style>
body {{ font-family: monospace; margin: 2em; }}
table {{ border-collapse: collapse; margin-bottom: 1em; }}
td, th {{ padding: 4px 8px; }}
details {{ margin: 4px 0; }}
summary {{ cursor: pointer; font-weight: bold; }}
</style>
</head>
<body>
<h1>torchtitan_npu Simulator Trace</h1>
{workload_section}
{rank_table_section}
<h2>L1/L0: Step Graphs</h2>
{step_sections}
</body>
</html>
"""


def render_html(workload_graph: WorkloadGraph) -> str:
    step_sections = "\n".join(_render_step_graph_section(sg) for sg in workload_graph.step_templates.values())
    return _PAGE_TEMPLATE.format(
        workload_section=_render_workload_section(workload_graph),
        rank_table_section=_render_rank_table_section(workload_graph),
        step_sections=step_sections,
    )


def export_html(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(workload_graph))
