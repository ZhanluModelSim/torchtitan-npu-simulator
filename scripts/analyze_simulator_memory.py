#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explain tensors retained across the forward-to-backward boundary.

Usage:
    python3 scripts/analyze_simulator_memory.py ./simulator_output/example

Pass either a simulator output directory (containing ``memory/``) or the
``memory/`` directory itself. The script is intentionally read-only and uses
only the exported CSV/JSON artifacts, so it can inspect results from another
machine or container.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _resolve_memory_dir(path: Path) -> Path:
    memory_dir = path / "memory" if (path / "memory").is_dir() else path
    required = ("memory_events.csv", "tensor_lifetimes.csv", "memory_summary.json")
    missing = [name for name in required if not (memory_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{memory_dir} is not a simulator memory export; missing: {', '.join(missing)}"
        )
    return memory_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _layer_name(module_path: str) -> str:
    match = _LAYER_PATTERN.search(module_path)
    return f"layer.{match.group(1)}" if match else "<outside-layers>"


def _gib(num_bytes: int) -> float:
    return num_bytes / 2**30


def analyze_memory_export(path: Path) -> dict[str, Any]:
    """Load an export and group forward tensors that survive into backward."""
    memory_dir = _resolve_memory_dir(path)
    raw_events = _read_csv(memory_dir / "memory_events.csv")
    lifetimes = _read_csv(memory_dir / "tensor_lifetimes.csv")
    summary = json.loads((memory_dir / "memory_summary.json").read_text(encoding="utf-8"))

    events_by_op = {row["op_id"]: row for row in raw_events}
    forward_seqs = [int(row["seq_idx"]) for row in raw_events if row["phase"] == "forward"]
    if not forward_seqs:
        raise ValueError("memory_events.csv contains no forward events")
    forward_end_seq = max(forward_seqs)

    residents: list[dict[str, Any]] = []
    for row in lifetimes:
        birth_seq = int(row["birth_seq"])
        death_seq = int(row["death_seq"])
        num_bytes = int(row["num_bytes"])
        if (
            row["producer_phase"] != "forward"
            or num_bytes <= 0
            or birth_seq > forward_end_seq
            or death_seq <= forward_end_seq
        ):
            continue
        producer = events_by_op.get(row["producer_op"], {})
        residents.append({
            **row,
            "num_bytes": num_bytes,
            "module_path": producer.get("module_path", ""),
            "layer": _layer_name(producer.get("module_path", "")),
        })

    by_kind: dict[str, int] = defaultdict(int)
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    suspicious_checkpoint_rows: list[dict[str, Any]] = []
    for row in residents:
        by_kind[row["kind"]] += row["num_bytes"]
        key = (row["layer"], row["kind"], row["shape"], row["dtype"], row["reason"])
        group = groups.setdefault(
            key,
            {
                "layer": row["layer"],
                "kind": row["kind"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "reason": row["reason"],
                "count": 0,
                "num_bytes": 0,
                "raw_ops": set(),
            },
        )
        group["count"] += 1
        group["num_bytes"] += row["num_bytes"]
        group["raw_ops"].add(row["producer_raw_op"])
        if row["kind"] == "activation" and "._checkpoint_wrapped_module" in row["module_path"]:
            suspicious_checkpoint_rows.append(row)

    ranked_groups = sorted(groups.values(), key=lambda item: item["num_bytes"], reverse=True)
    for group in ranked_groups:
        group["raw_ops"] = sorted(group["raw_ops"])
    return {
        "memory_dir": memory_dir,
        "summary": summary,
        "forward_end_seq": forward_end_seq,
        "resident_count": len(residents),
        "resident_bytes": sum(row["num_bytes"] for row in residents),
        "by_kind": dict(sorted(by_kind.items(), key=lambda item: item[1], reverse=True)),
        "groups": ranked_groups,
        "suspicious_checkpoint_count": len(suspicious_checkpoint_rows),
        "suspicious_checkpoint_bytes": sum(row["num_bytes"] for row in suspicious_checkpoint_rows),
    }


def print_report(analysis: dict[str, Any], top: int) -> None:
    """Print a compact, copyable diagnosis for a simulator memory export."""
    summary = analysis["summary"]
    print("Simulator memory forward-residency diagnosis")
    print(f"  export={analysis['memory_dir']}")
    print(f"  forward_end_seq={analysis['forward_end_seq']}")
    print(f"  persistent_param_bytes={_gib(int(summary['persistent_param_bytes'])):.3f} GiB")
    print(f"  active_bytes_peak={_gib(int(summary['active_bytes_peak'])):.3f} GiB")
    print(
        "  forward_to_backward_residents="
        f"{analysis['resident_count']} tensors, {_gib(analysis['resident_bytes']):.3f} GiB"
    )
    print("\nBy kind:")
    for kind, num_bytes in analysis["by_kind"].items():
        print(f"  {kind:<28} {_gib(num_bytes):8.3f} GiB")

    suspicious_count = analysis["suspicious_checkpoint_count"]
    suspicious_bytes = analysis["suspicious_checkpoint_bytes"]
    print(
        "\nCheckpoint-scope ordinary activations: "
        f"{suspicious_count} tensors, {_gib(suspicious_bytes):.3f} GiB"
    )
    if suspicious_count:
        print("  Inspect these first: they may be valid checkpoint-boundary outputs or missed internal releases.")

    print("\nLargest forward-to-backward residency groups:")
    if not analysis["groups"]:
        print("  <none>")
        return
    for group in analysis["groups"][:top]:
        each_mib = group["num_bytes"] / group["count"] / 2**20
        raw_ops = ",".join(group["raw_ops"][:2])
        print(
            f"  {group['layer']:<16} {group['kind']:<26} count={group['count']:<3} "
            f"total={_gib(group['num_bytes']):7.3f} GiB each={each_mib:7.1f} MiB\n"
            f"    shape={group['shape']} dtype={group['dtype']} reason={group['reason']} op={raw_ops}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Simulator output directory or its memory/ subdirectory")
    parser.add_argument("--top", type=int, default=80, help="Maximum grouped rows to print (default: 80)")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    print_report(analyze_memory_export(args.output_dir), args.top)


if __name__ == "__main__":
    main()
