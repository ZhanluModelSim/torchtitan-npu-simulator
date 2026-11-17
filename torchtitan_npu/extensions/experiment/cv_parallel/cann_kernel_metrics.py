# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Join CANN per-launch counters to CV-chunk trace tasks."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def classify_mixed_core_times(metrics: list[dict[str, Any]]) -> str | None:
    """Classify this launch/group by counters, never by an offline name table.

    Dominance is a scheduling approximation, not a pure-core guarantee. Use
    the same rule for FX resource attribution and native overlap reporting.
    """
    ratio_threshold, minor_max_ms = 0.75, 0.1
    aic_us = aiv_us = 0.0
    for counters in metrics:
        aic_value = counters.get("aicore_time(us)", counters.get("aic_time(us)"))
        aiv_value = counters.get("aiv_time(us)")
        if aic_value is None or aiv_value is None:
            return None
        try:
            aic, aiv = float(aic_value), float(aiv_value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) and value >= 0 for value in (aic, aiv)) or aic + aiv <= 0:
            return None
        # Prefer execution units to whole-core times that also include transfers.
        try:
            mac = float(counters["aic_mac_time(us)"])
            vec = float(counters["aiv_vec_time(us)"])
        except (KeyError, TypeError, ValueError):
            mac = vec = float("nan")
        if all(math.isfinite(value) and value >= 0 for value in (mac, vec)) and mac + vec > 0:
            aic, aiv = mac, vec
        aic_us += aic
        aiv_us += aiv
    total = aic_us + aiv_us
    if total > 0 and min(aic_us, aiv_us) <= minor_max_ms * 1000:
        if aic_us / total >= ratio_threshold:
            return "mix_aic"
        if aiv_us / total >= ratio_threshold:
            return "mix_aiv"
    return None


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


class CannKernelMetrics:
    """Counters retain CANN column names, units and raw values.

    Trace and CSV timestamps share the CANN clock. Decimal preserves sub-us
    identity even for epoch timestamps too large for float precision. Never
    match by kernel name alone or average launches with different shapes.
    """

    def __init__(self, work_path: Path):
        self._rows: dict[tuple[str, Decimal], list[dict[str, Any]]] = defaultdict(list)
        # op_summary retains all counter fields even when kernel_details is
        # configured to export only its basic columns. Avoid reading both
        # representations of the same launch.
        paths = sorted(work_path.rglob("op_summary*.csv")) or sorted(work_path.rglob("kernel_details.csv"))
        for path in paths:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    name = row.get("Op Name", row.get("Name"))
                    start = _decimal(row.get("Task Start Time(us)", row.get("Start Time(us)")))
                    duration = _decimal(row.get("Task Duration(us)", row.get("Duration(us)")))
                    if not name or start is None or duration is None:
                        continue
                    self._rows[name, start.quantize(Decimal("0.001"))].append({"row": row, "duration": duration})

    def core_metrics(self, event: dict[str, Any]) -> dict[str, Any]:
        args = event.get("args", {})
        start = _decimal(event["ts"])
        duration = _decimal(event["dur"])
        if start is None or duration is None:
            return {}
        candidates = [
            entry
            for entry in self._rows.get((str(event.get("name")), start.quantize(Decimal("0.001"))), ())
            if abs(entry["duration"] - duration) <= Decimal("0.001")
        ]
        # When available, stream/task identity disambiguates simultaneous
        # launches. Multiple remaining rows (including different devices)
        # remain explicitly ambiguous rather than borrowing another counter.
        for csv_keys, trace_key in (
            (("Stream Id", "Stream ID"), "Physic Stream Id"),
            (("Task Id", "Task ID"), "Task Id"),
        ):
            if trace_key in args:
                candidates = [
                    entry
                    for entry in candidates
                    if all(entry["row"][key].strip() == str(args[trace_key]) for key in csv_keys if key in entry["row"])
                ]
        if len(candidates) != 1:
            return {}
        return {
            key: value
            for key, value in candidates[0]["row"].items()
            if key.lower().startswith(("aic_", "aiv_", "aicore_"))
        }
