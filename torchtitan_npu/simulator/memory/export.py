# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSV/JSON/trace exports for static memory plans."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from torchtitan_npu.simulator.memory.records import MemoryPlan

_TRACE_TS_SCALE_US = 1000
_PHASES = {"forward", "backward", "optimizer"}


def _trace_ts(seq_idx: int) -> int:
    return max(0, int(seq_idx) + 1) * _TRACE_TS_SCALE_US


def _shapes(refs: tuple) -> str:
    return ";".join("[" + ",".join(str(dim) for dim in ref.shape) + "]" for ref in refs)


def _build_phase_spans(plan: MemoryPlan) -> list[dict]:
    spans: list[dict] = []
    current_phase = ""
    start_seq = 0
    last_seq = 0

    for event in sorted(plan.timeline_events, key=lambda item: (item.seq_idx, item.action)):
        phase = event.phase if event.phase in _PHASES else ""
        if phase != current_phase:
            if current_phase:
                spans.append({
                    "name": current_phase,
                    "ph": "X",
                    "pid": 1,
                    "tid": 2,
                    "ts": _trace_ts(start_seq),
                    "dur": max(_TRACE_TS_SCALE_US, _trace_ts(last_seq - start_seq + 1)),
                    "args": {"phase": current_phase},
                })
            current_phase = phase
            start_seq = event.seq_idx
        last_seq = event.seq_idx

    if current_phase:
        spans.append({
            "name": current_phase,
            "ph": "X",
            "pid": 1,
            "tid": 2,
            "ts": _trace_ts(start_seq),
            "dur": max(_TRACE_TS_SCALE_US, _trace_ts(last_seq - start_seq + 1)),
            "args": {"phase": current_phase},
        })
    return spans


def _build_execution_kind_spans(plan: MemoryPlan) -> list[dict]:
    spans: list[dict] = []
    current_kind = ""
    start_seq = 0
    last_seq = 0

    for event in sorted(plan.raw_events, key=lambda item: item.seq_idx):
        kind = event.execution_kind
        if kind != current_kind:
            if current_kind:
                spans.append({
                    "name": current_kind,
                    "ph": "X",
                    "pid": 1,
                    "tid": 5,
                    "ts": _trace_ts(start_seq),
                    "dur": max(_TRACE_TS_SCALE_US, _trace_ts(last_seq - start_seq + 1)),
                    "args": {"execution_kind": current_kind},
                })
            current_kind = kind
            start_seq = event.seq_idx
        last_seq = event.seq_idx

    if current_kind:
        spans.append({
            "name": current_kind,
            "ph": "X",
            "pid": 1,
            "tid": 5,
            "ts": _trace_ts(start_seq),
            "dur": max(_TRACE_TS_SCALE_US, _trace_ts(last_seq - start_seq + 1)),
            "args": {"execution_kind": current_kind},
        })
    return spans


def _build_schedule_action_spans(plan: MemoryPlan) -> list[dict]:
    spans: list[dict] = []
    stages = sorted({span.stage for span in plan.action_spans if span.stage >= 0})
    for stage in stages:
        spans.append({
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": 100 + stage,
            "args": {"name": f"PP stage {stage} actions"},
        })
    for span in plan.action_spans:
        tid = 100 + span.stage if span.stage >= 0 else 99
        label = span.comp_type or span.action_type
        if span.microbatch >= 0:
            label = f"{label} mb{span.microbatch}"
        spans.append({
            "name": label,
            "ph": "X",
            "pid": 1,
            "tid": tid,
            "ts": _trace_ts(span.start_seq),
            "dur": max(_TRACE_TS_SCALE_US, _trace_ts(span.end_seq - span.start_seq + 1)),
            "args": {
                "action_id": span.action_id,
                "action_type": span.action_type,
                "stage": span.stage,
                "microbatch": span.microbatch,
                "comp_type": span.comp_type,
                "phase": span.phase,
            },
        })
    return spans


def memory_plan_to_chrome_trace(plan: MemoryPlan) -> dict:
    """Build a compact Chrome Trace / Perfetto JSON payload.

    The trace intentionally contains only coarse signals:
    - active tensor bytes as a counter track;
    - forward/backward/optimizer phase spans;
    - one peak marker.
    """
    trace_events: list[dict] = [
        {"name": "process_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "simulator memory"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "active tensor bytes"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "training phase"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name": "fsdp full-param bytes"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 4, "args": {"name": "gradient accumulator bytes"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 5, "args": {"name": "execution kind"}},
    ]
    trace_events.extend(_build_phase_spans(plan))
    trace_events.extend(_build_execution_kind_spans(plan))
    trace_events.extend(_build_schedule_action_spans(plan))
    active_by_kind: dict[str, int] = {}
    for event in sorted(plan.timeline_events, key=lambda item: (item.seq_idx, item.action, item.tensor_id)):
        delta = event.num_bytes if event.action == "alloc" else -event.num_bytes
        active_by_kind[event.kind] = active_by_kind.get(event.kind, 0) + delta
        trace_events.append({
            "name": "active_bytes",
            "ph": "C",
            "pid": 1,
            "tid": 1,
            "ts": _trace_ts(event.seq_idx),
            "args": {
                "active_bytes": event.active_bytes_after,
                "action": event.action,
                "kind": event.kind,
                "phase": event.phase,
            },
        })
        if event.kind == "fsdp_full_param":
            trace_events.append({
                "name": "active_fsdp_full_param_bytes",
                "ph": "C",
                "pid": 1,
                "tid": 3,
                "ts": _trace_ts(event.seq_idx),
                "args": {
                    "active_fsdp_full_param_bytes": active_by_kind.get("fsdp_full_param", 0),
                    "action": event.action,
                    "phase": event.phase,
                    "reason": event.reason,
                },
            })
        if event.kind == "gradient_accumulator":
            trace_events.append({
                "name": "active_gradient_accumulator_bytes",
                "ph": "C",
                "pid": 1,
                "tid": 4,
                "ts": _trace_ts(event.seq_idx),
                "args": {
                    "active_gradient_accumulator_bytes": active_by_kind.get("gradient_accumulator", 0),
                    "action": event.action,
                    "phase": event.phase,
                    "reason": event.reason,
                },
            })
    trace_events.append({
        "name": "peak active bytes",
        "ph": "i",
        "s": "g",
        "pid": 1,
        "tid": 1,
        "ts": _trace_ts(plan.peak_seq_idx),
        "args": {
            "active_bytes_peak": plan.peak_active_bytes,
            "peak_phase": plan.peak_phase,
        },
    })
    return {
        "displayTimeUnit": "ms",
        "traceEvents": trace_events,
        "metadata": {
            "metric": plan.metric,
            "persistent_param_bytes": plan.persistent_param_bytes,
            "active_bytes_peak": plan.peak_active_bytes,
            "forward_active_bytes_peak": plan.forward_peak_active_bytes,
            "backward_active_bytes_peak": plan.backward_peak_active_bytes,
            "optimizer_active_bytes_peak": plan.optimizer_peak_active_bytes,
            "peak_seq_idx": plan.peak_seq_idx,
            "memory_action_span_count": len(plan.action_spans),
        },
    }


def export_memory_plan(plan: MemoryPlan, out_dir: str) -> None:
    memory_dir = os.path.join(out_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    with open(os.path.join(memory_dir, "memory_summary.json"), "w", encoding="utf-8") as f:
        json.dump(plan.to_summary_dict(), f, indent=2, ensure_ascii=False)

    with open(os.path.join(memory_dir, "memory_trace.json"), "w", encoding="utf-8") as f:
        json.dump(memory_plan_to_chrome_trace(plan), f, indent=2, ensure_ascii=False)

    with open(os.path.join(memory_dir, "memory_events.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "event_id",
            "seq_idx",
            "phase",
            "execution_kind",
            "op_id",
            "raw_op_type",
            "op_type",
            "module_path",
            "input_bytes",
            "output_bytes",
            "input_shapes",
            "output_shapes",
            "pp_stage",
            "pp_mb_idx",
            "comp_type",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in plan.raw_events:
            writer.writerow({
                "event_id": event.event_id,
                "seq_idx": event.seq_idx,
                "phase": event.phase,
                "execution_kind": event.execution_kind,
                "op_id": event.op_id,
                "raw_op_type": event.raw_op_type,
                "op_type": event.op_type,
                "module_path": event.module_path,
                "input_bytes": sum(ref.num_bytes for ref in event.inputs),
                "output_bytes": sum(ref.num_bytes for ref in event.outputs),
                "input_shapes": _shapes(event.inputs),
                "output_shapes": _shapes(event.outputs),
                "pp_stage": event.pp_stage,
                "pp_mb_idx": event.pp_mb_idx,
                "comp_type": event.comp_type,
            })

    if plan.action_spans:
        with open(os.path.join(memory_dir, "memory_actions.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "action_id", "action_type", "stage", "microbatch", "comp_type",
                "phase", "start_seq", "end_seq", "source_seq_idx",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for span in plan.action_spans:
                writer.writerow(asdict(span))

    with open(os.path.join(memory_dir, "memory_timeline.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seq_idx",
            "phase",
            "op_id",
            "action",
            "tensor_id",
            "kind",
            "num_bytes",
            "active_bytes_after",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in plan.timeline_events:
            writer.writerow(asdict(event))

    with open(os.path.join(memory_dir, "tensor_lifetimes.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tensor_id",
            "kind",
            "num_bytes",
            "birth_seq",
            "death_seq",
            "producer_op",
            "producer_raw_op",
            "producer_phase",
            "consumer_ops",
            "consumer_seqs",
            "consumer_phases",
            "alias_of",
            "shape",
            "dtype",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lifetime in plan.tensor_lifetimes:
            row = asdict(lifetime)
            row["consumer_ops"] = ";".join(str(item) for item in lifetime.consumer_ops)
            row["consumer_seqs"] = ";".join(str(item) for item in lifetime.consumer_seqs)
            row["consumer_phases"] = ";".join(lifetime.consumer_phases)
            row["shape"] = "[" + ",".join(str(dim) for dim in lifetime.shape) + "]"
            writer.writerow(row)

    if plan.unclassified_ops:
        with open(os.path.join(memory_dir, "unclassified_memory_ops.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = ["seq_idx", "op_id", "raw_op_type", "phase", "output_bytes"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in plan.unclassified_ops:
                writer.writerow(item)
