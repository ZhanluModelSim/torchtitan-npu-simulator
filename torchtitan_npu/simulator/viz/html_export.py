# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained HTML visualization of the captured four-layer IR.

Renders an interactive, collapsible tree that mirrors the L3→L2→L1→L0
hierarchy with:
- L3: WorkloadGraph header
- L2: ScheduleGraph with parallel strategy, RankTable, comm stats,
  schedule timeline (PP/FSDP/fwd-bwd-opt ordering), and DataPasses
- L1: StepGraph templates with L0 op tables (repeated layers merged)
- L0: OpNode rows in topological order, with shapes/flops/comm info
"""

from __future__ import annotations

import html
import re
from collections import deque

from torchtitan_npu.simulator.capture.op_mapping import display_op_label
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph

_NUMERIC_SEGMENT = re.compile(r"\.\d+(?=\.|$)")


def _normalize_module_path(path: str) -> str:
    if not path:
        return path
    return _NUMERIC_SEGMENT.sub(".N", path)


def _topo_sort(nodes: dict[int, OpNode]) -> list[int]:
    in_degree = {op_id: sum(1 for p in node.predecessors if p in nodes) for op_id, node in nodes.items()}
    ready = sorted(op_id for op_id, deg in in_degree.items() if deg == 0)
    queue: deque[int] = deque(ready)
    result: list[int] = []
    while queue:
        op_id = queue.popleft()
        result.append(op_id)
        newly_ready = []
        for succ in nodes[op_id].successors:
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    newly_ready.append(succ)
        if newly_ready:
            queue = deque(sorted(list(queue) + sorted(newly_ready)))
    return result


def _shapes_str(metas: list) -> str:
    return ";".join("[" + ",".join(str(d) for d in m.shape) + "]" for m in metas)


def _fmt_bytes(n: int) -> str:
    if n == 0:
        return "0"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(f) < 1024:
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}EB"


def _fmt_flops(n: int) -> str:
    if n == 0:
        return "0"
    f = float(n)
    for unit in ("", "K", "M", "G", "T", "P"):
        if abs(f) < 1000:
            return f"{f:.1f}{unit}"
        f /= 1000
    return f"{f:.1f}E"


def _is_comm_op(ann: dict) -> bool:
    return ann.get("raw_op_type", "").startswith("comm.")


def _render_l0_op_row(op_id: int, node: OpNode, topo_idx: int) -> str:
    ann = node.annotations
    label = display_op_label(node.op_type, ann)
    raw = ann.get("raw_op_type", "")
    is_comm = _is_comm_op(ann)
    cls = "comm-row" if is_comm else ""
    comm_dim = ann.get("comm_dim", "")
    comm_ranks = ann.get("comm_ranks", "")
    repeat = ann.get("repeat_count", 1)
    module_path = ann.get("module_path", "")

    return (
        f"<tr class='{cls}'>"
        f"<td class='num'>{topo_idx}</td>"
        f"<td class='num'>{op_id}</td>"
        f"<td class='op-type'>{html.escape(label)}</td>"
        f"<td class='raw'>{html.escape(raw)}</td>"
        f"<td class='mono'>{html.escape(_shapes_str(node.inputs))}</td>"
        f"<td class='mono'>{html.escape(_shapes_str(node.outputs))}</td>"
        f"<td class='num'>{_fmt_flops(node.flops)}</td>"
        f"<td class='num'>{_fmt_bytes(node.peak_mem)}</td>"
        f"<td class='num'>{_fmt_bytes(node.comm_bytes)}</td>"
        f"<td class='num'>{repeat}</td>"
        f"<td class='path'>{html.escape(module_path)}</td>"
        f"<td>{html.escape(comm_dim)}</td>"
        f"<td class='ranks'>{html.escape(comm_ranks)}</td>"
        f"</tr>"
    )


def _render_l1_step_graph(step_graph: StepGraph) -> str:
    sorted_ids = _topo_sort(step_graph.nodes)
    total_flops = sum(n.flops for n in step_graph.nodes.values())
    total_comm = sum(n.comm_bytes for n in step_graph.nodes.values())
    total_peak = sum(n.peak_mem for n in step_graph.nodes.values())
    comm_count = sum(1 for n in step_graph.nodes.values() if _is_comm_op(n.annotations))

    # Group consecutive ops by normalized module path to merge repeated layers
    groups: list[tuple[str, int, int, list[tuple[int, OpNode]]]] = []  # (norm_path, start_topo, count, ops)
    current_group: list[tuple[int, OpNode]] = []
    current_norm: str = ""
    group_start: int = 0

    for idx, op_id in enumerate(sorted_ids):
        node = step_graph.nodes[op_id]
        raw_path = node.annotations.get("module_path", "")
        norm = _normalize_module_path(raw_path)

        if norm != current_norm and current_group:
            groups.append((current_norm, group_start, len(current_group), current_group))
            current_group = []
            group_start = idx
        elif not current_group:
            group_start = idx

        current_norm = norm
        current_group.append((op_id, node))

    if current_group:
        groups.append((current_norm, group_start, len(current_group), current_group))

    # Render groups: merge consecutive groups with same norm_path
    merged_groups: list[tuple[str, int, int, list[tuple[int, OpNode]]]] = []
    for norm, start, count, ops in groups:
        if merged_groups and merged_groups[-1][0] == norm:
            # Merge with previous group
            prev_norm, prev_start, prev_count, prev_ops = merged_groups[-1]
            merged_groups[-1] = (prev_norm, prev_start, prev_count + count, prev_ops + ops)
        else:
            merged_groups.append((norm, start, count, ops))

    # Render each merged group
    group_html_parts = []
    for norm, start, count, ops in merged_groups:
        # Determine if this is a repeated layer group
        is_repeated = count > 1 and norm != "" and norm != "(root)"
        if is_repeated:
            # Show first op's full path and indicate repetition
            first_path = ops[0][1].annotations.get("module_path", "")
            last_path = ops[-1][1].annotations.get("module_path", "")
            label = html.escape(norm)
            detail = f"<div class='group-header'>📋 {label} <span class='repeat-badge'>×{count} ops</span> <span class='path-range'>[{first_path} ... {last_path}]</span></div>"
        else:
            detail = f"<div class='group-header'>{html.escape(norm) if norm else '(root)'}</div>"

        rows = "".join(_render_l0_op_row(op_id, node, start + i) for i, (op_id, node) in enumerate(ops))
        group_html_parts.append(detail + f"<table class='op-table'><tbody>{rows}</tbody></table>")

    groups_html = "\n".join(group_html_parts)

    return f"""
<details class="l1" open>
<summary>
  <span class="badge l1-badge">L1</span>
  <strong>{html.escape(step_graph.step_type)}</strong>
  <span class="meta">{len(step_graph.nodes)} ops ({comm_count} comm) &middot;
  flops={_fmt_flops(total_flops)} &middot; peak_mem={_fmt_bytes(total_peak)} &middot;
  comm={_fmt_bytes(total_comm)} &middot; acyclic={step_graph.is_acyclic}</span>
</summary>
<div class="op-groups">
<div class="op-table-header">
<span>#</span><span>op_id</span><span>op_type</span><span>raw_op_type</span>
<span>inputs</span><span>outputs</span>
<span>flops</span><span>peak_mem</span><span>comm_bytes</span>
<span>repeat</span><span>module_path</span><span>comm_dim</span><span>comm_ranks</span>
</div>
{groups_html}
</div>
</details>"""


def _render_l2_schedule_timeline(schedule) -> str:
    """Render execution timeline from captured data (not inferred).

    Shows the actual execution order of ops (by seq_idx), with PP
    stage/microbatch context for P2P communication ops.  Each row is
    one TimelineEntry from the captured execution_timeline."""
    timeline = schedule.execution_timeline
    if not timeline:
        return "<p><em>No execution timeline captured.</em></p>"

    # Group timeline entries by phase for a summary
    phase_counts: dict[str, int] = {}
    p2p_count = 0
    for entry in timeline:
        if entry.comm_type:
            phase_counts["comm"] = phase_counts.get("comm", 0) + 1
            if entry.comm_type.startswith("fwd") or entry.comm_type.startswith("bwd"):
                p2p_count += 1
        else:
            phase_counts[entry.phase] = phase_counts.get(entry.phase, 0) + 1

    summary = " &middot; ".join(f"{k}: {v}" for k, v in sorted(phase_counts.items()))
    if p2p_count:
        summary += f" &middot; P2P: {p2p_count}"

    # Render P2P events as a timeline table (these show PP scheduling)
    p2p_entries = [e for e in timeline if e.comm_type and ("send" in e.comm_type or "recv" in e.comm_type)]
    if p2p_entries:
        p2p_rows = ""
        for e in p2p_entries[:200]:  # limit for performance
            cls = "cell-fwd" if "forward" in e.comm_type else "cell-bwd"
            p2p_rows += (
                f"<tr class='{cls}'>"
                f"<td class='num'>{e.seq_idx}</td>"
                f"<td class='num'>{e.op_id}</td>"
                f"<td>{html.escape(e.comm_type)}</td>"
                f"<td class='num'>{e.pipeline_stage}</td>"
                f"<td class='num'>{e.micro_batch_idx}</td>"
                f"<td class='num'>{e.comm_peer_rank}</td>"
                f"</tr>"
            )
        more = f"<p>... and {len(p2p_entries) - 200} more</p>" if len(p2p_entries) > 200 else ""
        p2p_table = f"""
<h3>Pipeline P2P Communication Timeline (captured)</h3>
<p class="hint">Each row = one P2P send/recv call, ordered by actual execution sequence (seq_idx). Stage and microbatch come from captured PP context.</p>
<table class="timeline-table">
<thead><tr><th>seq_idx</th><th>op_id</th><th>comm_type</th>
<th>PP stage</th><th>microbatch</th><th>peer_rank</th></tr></thead>
<tbody>{p2p_rows}</tbody>
</table>{more}"""
    else:
        p2p_table = "<p><em>No P2P communication captured (PP=1 or no pipeline schedule).</em></p>"

    # Render collective comm events timeline
    coll_entries = [e for e in timeline if e.comm_type and "send" not in e.comm_type and "recv" not in e.comm_type]
    if coll_entries:
        coll_rows = ""
        for e in coll_entries[:100]:
            coll_rows += (
                f"<tr>"
                f"<td class='num'>{e.seq_idx}</td>"
                f"<td class='num'>{e.op_id}</td>"
                f"<td>{html.escape(e.comm_type)}</td>"
                f"</tr>"
            )
        more = f"<p>... and {len(coll_entries) - 100} more</p>" if len(coll_entries) > 100 else ""
        coll_table = f"""
<h3>Collective Communication Timeline (captured)</h3>
<table class="timeline-table">
<thead><tr><th>seq_idx</th><th>op_id</th><th>comm_type</th></tr></thead>
<tbody>{coll_rows}</tbody>
</table>{more}"""
    else:
        coll_table = ""

    return f"""
<h3>Execution Timeline (captured, {len(timeline)} entries)</h3>
<p class="meta">{summary}</p>
{p2p_table}
{coll_table}"""


def _render_l2_schedule(workload_graph: WorkloadGraph) -> str:
    schedule = workload_graph.iteration.schedule
    rank_table = schedule.annotations.get("rank_table", {}) if schedule.annotations else {}

    stage_counts: dict[int, int] = {}
    for inst in schedule.instances:
        stage_counts[inst.pipeline_stage] = stage_counts.get(inst.pipeline_stage, 0) + 1
    stage_summary = ", ".join(f"stage{s}: {c} ranks" for s, c in sorted(stage_counts.items()))

    comm_stats: dict[str, int] = {}
    for dp in schedule.data_passes:
        vol = sum(s.volume_bytes for s in dp.slots)
        comm_stats[dp.comm_primitive] = comm_stats.get(dp.comm_primitive, 0) + vol

    comm_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{_fmt_bytes(v)}</td></tr>"
        for k, v in sorted(comm_stats.items())
    )

    # DataPasses: show what they are and link to L0
    dp_sample = schedule.data_passes[:50]
    dp_rows = ""
    for dp in dp_sample:
        slot_info = "; ".join(
            f"{html.escape(s.name)}: {html.escape(str(s.shape))} {_fmt_bytes(s.volume_bytes)}"
            for s in dp.slots
        )
        src_op = dp.slots[0].src_exit_op if dp.slots else 0
        dst_op = dp.slots[0].dst_entry_op if dp.slots else 0
        dp_rows += (
            f"<tr>"
            f"<td>{html.escape(dp.src_instance)}</td>"
            f"<td>{html.escape(dp.dst_instance)}</td>"
            f"<td>{html.escape(dp.comm_primitive)}</td>"
            f"<td class='num'>{src_op}</td>"
            f"<td class='num'>{dst_op}</td>"
            f"<td class='mono'>{slot_info}</td>"
            f"</tr>"
        )
    dp_more = f"<p>... and {len(schedule.data_passes) - len(dp_sample)} more</p>" if len(schedule.data_passes) > 50 else ""

    # RankTable
    dim_degrees = rank_table.get("dim_degrees", {})
    process_groups = rank_table.get("process_groups", {})
    rt_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td>"
        f"<td>{len(process_groups.get(k, []))} groups</td></tr>"
        for k, v in sorted(dim_degrees.items())
    )
    group_details = ""
    for dim, groups in sorted(process_groups.items()):
        if dim_degrees.get(dim, 1) <= 1:
            continue
        sample = groups[:4]
        group_strs = "; ".join("[" + ",".join(str(r) for r in g) + "]" for g in sample)
        more = f" ... (+{len(groups)-4} more)" if len(groups) > 4 else ""
        group_details += f"<div><strong>{html.escape(dim)}</strong>: {group_strs}{more}</div>"

    return f"""
<details class="l2" open>
<summary>
  <span class="badge l2-badge">L2</span>
  <strong>ScheduleGraph</strong>
  <span class="meta">
    {len(schedule.instances)} instances &middot; {len(schedule.data_passes)} data_passes &middot;
    dp={schedule.dp_degree} tp={schedule.tp_degree} pp={schedule.pp_degree} &middot;
    {html.escape(schedule.pipeline_schedule)} &middot; {stage_summary}
  </span>
</summary>

<h3>Parallel Strategy</h3>
<table class="info-table">
<tr><th>dp_degree</th><td>{schedule.dp_degree}</td>
    <th>tp_degree</th><td>{schedule.tp_degree}</td>
    <th>pp_degree</th><td>{schedule.pp_degree}</td></tr>
<tr><th>num_micro_batches</th><td>{schedule.num_micro_batches}</td>
    <th>gradient_accumulation</th><td>{schedule.gradient_accumulation}</td>
    <th>pipeline_schedule</th><td>{html.escape(schedule.pipeline_schedule)}</td></tr>
</table>

{_render_l2_schedule_timeline(schedule)}

<h3>RankTable (Communication Domains)</h3>
<table class="rt-table">
<thead><tr><th>dimension</th><th>degree</th><th>#groups</th></tr></thead>
<tbody>{rt_rows}</tbody>
</table>
<div class="group-details">{group_details}</div>

<h3>Communication Statistics</h3>
<table class="comm-table">
<thead><tr><th>primitive</th><th>total_bytes</th></tr></thead>
<tbody>{comm_rows}</tbody>
</table>

<h3>DataPasses (Cross-Rank Communication)</h3>
<p class="hint">Each row = one tensor transfer between two ranks. <code>src_exit_op</code>/<code>dst_entry_op</code> are L0 OpNode IDs that produce/consume the tensor. PP P2P communication (isend/irecv) appears here as <code>comm_primitive=</code> entries when pipeline parallelism is active.</p>
<table class="dp-table">
<thead><tr><th>src_instance</th><th>dst_instance</th><th>comm_primitive</th>
<th>src_exit_op (L0)</th><th>dst_entry_op (L0)</th><th>slots</th></tr></thead>
<tbody>{dp_rows}</tbody>
</table>{dp_more}
</details>"""


def _render_l3_workload(workload_graph: WorkloadGraph) -> str:
    inputs = "".join(
        f"<li><strong>{html.escape(f.source)}</strong>: shape={html.escape(str(f.tensor_shape))} "
        f"dtype={html.escape(f.dtype)} volume={_fmt_bytes(f.volume_per_iter)}</li>"
        for f in workload_graph.data_inputs
    )
    return f"""
<details class="l3" open>
<summary>
  <span class="badge l3-badge">L3</span>
  <strong>WorkloadGraph</strong>
  <span class="meta">
    id={html.escape(workload_graph.workload_id)} &middot; type={html.escape(workload_graph.workload_type)} &middot;
    iterations={workload_graph.num_iterations} (warmup={workload_graph.warmup_iterations})
  </span>
</summary>
<table class="info-table">
<tr><th>workload_id</th><td>{html.escape(workload_graph.workload_id)}</td>
    <th>workload_type</th><td>{html.escape(workload_graph.workload_type)}</td></tr>
<tr><th>num_iterations</th><td>{workload_graph.num_iterations}</td>
    <th>warmup_iterations</th><td>{workload_graph.warmup_iterations}</td></tr>
<tr><th>microbatch_count</th><td>{workload_graph.iteration.microbatch_count}</td>
    <th>step_templates</th><td>{', '.join(html.escape(k) for k in workload_graph.step_templates)}</td></tr>
</table>
<h3>Data Inputs</h3>
<ul>{inputs}</ul>
</details>"""


_CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #1a1a1a; }
.header { background: #1a1a2e; color: #fff; padding: 16px 24px; position: sticky; top: 0; z-index: 100; }
.header h1 { margin: 0; font-size: 18px; }
.header .controls { float: right; }
.header button { background: #4a4a6a; color: #fff; border: 1px solid #6a6a8a; padding: 4px 12px; cursor: pointer; border-radius: 3px; font-size: 12px; margin-left: 4px; }
.header button:hover { background: #5a5a7a; }
.header input { background: #2a2a3e; color: #fff; border: 1px solid #4a4a6a; padding: 4px 8px; border-radius: 3px; font-size: 12px; width: 200px; }
.container { padding: 16px 24px; }
details { margin: 4px 0; border-radius: 4px; }
details.l3 { background: #e8f0fe; border: 1px solid #c2d6f0; }
details.l2 { background: #fef3e8; border: 1px solid #f0d6c2; margin-left: 16px; }
details.l1 { background: #e8feee; border: 1px solid #c2f0d0; margin-left: 32px; }
summary { cursor: pointer; padding: 8px 12px; font-size: 14px; user-select: none; }
summary:hover { background: rgba(0,0,0,0.05); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; color: #fff; margin-right: 6px; }
.l3-badge { background: #4285f4; }
.l2-badge { background: #f5a623; }
.l1-badge { background: #34a853; }
.meta { color: #666; font-size: 12px; margin-left: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0; }
th, td { padding: 3px 6px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f0f0f0; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.mono, th.mono { font-family: 'SF Mono', Consolas, monospace; font-size: 11px; }
td.op-type { font-weight: 600; }
td.raw { color: #888; font-size: 11px; }
td.path { color: #555; font-size: 11px; }
td.ranks { color: #0066cc; font-size: 11px; }
.comm-row { background: #fff3e0 !important; }
.comm-row td.op-type { color: #e65100; }
.info-table th { width: 120px; }
.op-groups { margin: 4px 0; }
.group-header { font-size: 12px; color: #333; margin: 8px 0 2px; padding: 4px 8px; background: #e8e8e8; border-radius: 3px; }
.repeat-badge { background: #4285f4; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
.path-range { color: #888; font-size: 11px; margin-left: 8px; }
.op-table { table-layout: fixed; margin: 0; }
.op-table th, .op-table td { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.op-table-header { display: flex; font-size: 11px; font-weight: 600; color: #666; padding: 4px 6px; border-bottom: 2px solid #ccc; }
.op-table-header span { flex: 1; }
.op-table-header span:nth-child(1) { flex: 0 0 40px; }
.op-table-header span:nth-child(2) { flex: 0 0 60px; }
.op-table-header span:nth-child(3) { flex: 0 0 120px; }
.op-table-header span:nth-child(4) { flex: 0 0 180px; }
.op-table-header span:nth-child(5), .op-table-header span:nth-child(6) { flex: 0 0 150px; }
.op-table-header span:nth-child(7), .op-table-header span:nth-child(8), .op-table-header span:nth-child(9) { flex: 0 0 80px; }
.op-table-header span:nth-child(10) { flex: 0 0 50px; }
.op-table-header span:nth-child(11) { flex: 0 0 200px; }
.op-table-header span:nth-child(12) { flex: 0 0 80px; }
.op-table-header span:nth-child(13) { flex: 0 0 150px; }
.op-table th:nth-child(1) { width: 40px; }
.op-table th:nth-child(2) { width: 60px; }
.op-table th:nth-child(3) { width: 120px; }
.op-table th:nth-child(4) { width: 180px; }
.op-table th:nth-child(5), .op-table th:nth-child(6) { width: 150px; }
.op-table th:nth-child(7), .op-table th:nth-child(8), .op-table th:nth-child(9) { width: 80px; }
.op-table th:nth-child(10) { width: 50px; }
.op-table th:nth-child(11) { width: 200px; }
.op-table th:nth-child(12) { width: 80px; }
.op-table th:nth-child(13) { width: 150px; }
.dp-table code { font-size: 11px; }
.rt-table th:nth-child(1) { width: 120px; }
.rt-table th:nth-child(2) { width: 60px; }
.group-details { margin: 8px 0; font-size: 12px; }
.group-details div { margin: 2px 0; }
.comm-table th:nth-child(1) { width: 150px; }
.timeline-table { font-size: 11px; }
.timeline-table th, .timeline-table td { text-align: center; width: 40px; }
.cell-fwd { background: #d4edda; color: #155724; }
.cell-bwd { background: #f8d7da; color: #721c24; }
.cell-fb { background: #fff3cd; color: #856404; }
.cell-idle { background: #f0f0f0; color: #ccc; }
.timeline-legend { margin: 4px 0; font-size: 11px; }
.timeline-legend span { display: inline-block; padding: 2px 8px; margin-right: 8px; border-radius: 3px; }
.hint { font-size: 11px; color: #666; margin: 4px 0; }
h3 { font-size: 13px; margin: 12px 0 4px; color: #333; }
ul { margin: 4px 0; padding-left: 20px; font-size: 12px; }
.hidden { display: none !important; }
"""

_JS = """
function toggleAll(open) {
  document.querySelectorAll('details').forEach(d => d.open = open);
}
function filterOps() {
  var q = document.getElementById('op-filter').value.toLowerCase();
  document.querySelectorAll('.op-table tbody tr').forEach(function(tr) {
    var text = tr.textContent.toLowerCase();
    tr.classList.toggle('hidden', q && !text.includes(q));
  });
}
"""


def render_html(workload_graph: WorkloadGraph) -> str:
    l3 = _render_l3_workload(workload_graph)
    l2 = _render_l2_schedule(workload_graph)
    l1_sections = "\n".join(
        _render_l1_step_graph(sg) for sg in workload_graph.step_templates.values()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>torchtitan_npu Simulator IR Trace</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>torchtitan_npu Simulator &mdash; Four-Layer IR Trace</h1>
  <div class="controls">
    <input id="op-filter" type="text" placeholder="Filter ops..." oninput="filterOps()">
    <button onclick="toggleAll(true)">Expand All</button>
    <button onclick="toggleAll(false)">Collapse All</button>
  </div>
</div>
<div class="container">
{l3}
{l2}
<h2 style="margin-left:32px;">L1 StepGraphs &rarr; L0 OpNodes</h2>
{l1_sections}
</div>
<script>{_JS}</script>
</body>
</html>"""


def export_html(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(workload_graph))
