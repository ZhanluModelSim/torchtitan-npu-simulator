# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure CV-chunk Cube/Vector overlap in a detailed CANN capture."""

import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from .cann_kernel_metrics import CannKernelMetrics, classify_mixed_core_times

SCOPE = "NPU::GraphTrainer::forward_backward"
CORE_TYPES = {
    "AI_CORE": "cube",
    "AICORE": "cube",
    "AIC": "cube",
    "CUBE": "cube",
    "AI_VECTOR_CORE": "vector",
    "VECTOR_CORE": "vector",
    "AIVEC": "vector",
    "AIV": "vector",
}


def _ns(value):
    number = Decimal(str(value)) * 1000
    if not number.is_finite():
        raise ValueError(f"Invalid profiler timestamp: {value}")
    return int(number)


def summarize_profile(output: Path, *, trace=None, kernel_metrics=None):
    """Summarize F/B compute intervals; MIX never contributes to pure CV."""
    output = output.resolve()
    if trace is None:
        trace = json.loads((output / "trace_view.json").read_text(), parse_float=Decimal)
    trace = trace if isinstance(trace, list) else trace["traceEvents"]
    windows = []
    for event in sorted(
        (e for e in trace if e.get("ph") == "X" and e.get("name") == SCOPE), key=lambda e: _ns(e["ts"])
    ):
        start, duration = _ns(event["ts"]), _ns(event["dur"])
        if duration <= 0:
            raise ValueError("Forward/backward scope has no duration")
        end = start + duration
        if windows and start <= windows[-1][1]:
            windows[-1] = windows[-1][0], max(end, windows[-1][1])
        else:
            windows.append((start, end))
    if not windows:
        raise ValueError(f"Missing {SCOPE} scope in {output}")

    kernels, devices, counts = [], set(), Counter()
    mixed_counts = Counter()
    with (output / "kernel_details.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Device_id", "Stream ID", "Accelerator Core", "Start Time(us)", "Duration(us)"}
        if missing := required - set(reader.fieldnames or []):
            raise ValueError(f"Missing detailed profiler columns: {sorted(missing)}")
        for row in reader:
            start, duration = _ns(row["Start Time(us)"]), _ns(row["Duration(us)"])
            if duration < 0:
                raise ValueError("Negative kernel duration")
            core = row["Accelerator Core"].strip().upper()
            kind = "mix" if "MIX" in core else CORE_TYPES.get(core)
            description = (row.get("Name", "") + " " + row.get("Type", "")).upper()
            if kind is None or any(word in description for word in ("HCCL", "HCOM", "COMMUNICATION")):
                continue
            intervals = [
                (max(start, begin), min(start + duration, end))
                for begin, end in windows
                if min(start + duration, end) > max(start, begin)
            ]
            if not intervals:
                continue
            devices.add(row["Device_id"])
            counts[kind] += 1
            if kind == "mix":
                if kernel_metrics is None:
                    kernel_metrics = CannKernelMetrics(output.parent)
                counters = kernel_metrics.core_metrics(
                    {
                        "name": row.get("Name"),
                        "ts": row["Start Time(us)"],
                        "dur": row["Duration(us)"],
                        "args": {"Physic Stream Id": row["Stream ID"]},
                    }
                )
                kind = classify_mixed_core_times([counters]) or "mix"
                mixed_counts[kind] += 1
            kernels.extend((begin, end, row["Stream ID"], kind) for begin, end in intervals)
    if len(devices) != 1 or not kernels:
        raise ValueError(f"Expected compute kernels on one captured device, found {sorted(devices)}")
    streams = sorted({stream for _, _, stream, _ in kernels})
    if not 1 <= len(streams) <= 2:
        raise ValueError(f"Expected one or two compute streams, found {streams}")

    events = defaultdict(list)
    for start, end, stream, kind in kernels:
        events[start].append((stream, kind, 1))
        events[end].append((stream, kind, -1))
    active = {stream: Counter() for stream in streams}
    totals = Counter()
    stream_busy = Counter()
    previous = min(events)
    for timestamp in sorted(events):
        duration = timestamp - previous
        occupied = [
            {kind for kind, count in active[stream].items() if count > 0}
            for stream in streams
            if any(count > 0 for count in active[stream].values())
        ]
        if occupied:
            totals["compute_union"] += duration
            for stream in streams:
                if any(count > 0 for count in active[stream].values()):
                    stream_busy[stream] += duration
            for kind in ("cube", "vector", "mix"):
                if any(any(value.split("_")[0] == kind for value in kinds) for kinds in occupied):
                    totals[kind + "_busy"] += duration
            if len(occupied) == 2:
                totals["overlap"] += duration
                pair = (
                    {next(iter(kinds)) for kinds in occupied} if all(len(kinds) == 1 for kinds in occupied) else set()
                )
                if pair == {"cube", "mix_aiv"}:
                    totals["cube_mix_aiv_overlap"] += duration
                elif pair == {"vector", "mix_aic"}:
                    totals["vector_mix_aic_overlap"] += duration
                if any(len(kinds) != 1 or any(kind.startswith("mix") for kind in kinds) for kinds in occupied):
                    category = "mixed_overlap"
                elif occupied[0] != occupied[1]:
                    category = "cv_overlap"
                else:
                    category = "cc_overlap" if occupied[0] == {"cube"} else "vv_overlap"
                totals[category] += duration
            else:
                totals["single_stream"] += duration
        for stream, kind, delta in events[timestamp]:
            active[stream][kind] += delta
        previous = timestamp
    totals["compute_span"] = max(events) - min(events)
    totals["no_compute_in_span"] = totals["compute_span"] - totals["compute_union"]
    totals["forward_backward_scope"] = sum(end - start for start, end in windows)
    totals["kernel_duration_sum"] = sum(end - start for start, end, _, _ in kernels)
    keys = (
        "forward_backward_scope",
        "compute_span",
        "compute_union",
        "kernel_duration_sum",
        "single_stream",
        "overlap",
        "cv_overlap",
        "cube_mix_aiv_overlap",
        "vector_mix_aic_overlap",
        "cc_overlap",
        "vv_overlap",
        "mixed_overlap",
        "cube_busy",
        "vector_busy",
        "mix_busy",
        "no_compute_in_span",
    )
    return {
        "device": next(iter(devices)),
        "streams": streams,
        "kernel_counts": dict(counts),
        "mixed_kernel_counts": dict(mixed_counts),
        "time_ms": {key: totals[key] / 1_000_000 for key in keys},
        "stream_busy_ms": {stream: stream_busy[stream] / 1_000_000 for stream in streams},
        "overlap_percent": 100 * totals["overlap"] / totals["compute_union"],
        "cv_percent_of_compute": 100 * totals["cv_overlap"] / totals["compute_union"],
        "cube_covered_by_vector_percent": 100 * totals["cv_overlap"] / totals["cube_busy"]
        if totals["cube_busy"]
        else None,
        "vector_covered_by_cube_percent": 100 * totals["cv_overlap"] / totals["vector_busy"]
        if totals["vector_busy"]
        else None,
    }
