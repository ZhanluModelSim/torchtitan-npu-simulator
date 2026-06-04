#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Compare loss and grad_norm between two training stdout log files.

Parses torchtitan-npu training logs (the same format consumed by
training-log-visualization) and checks whether loss and grad_norm are
bit-wise identical (or within floating-point tolerance).

Usage:
    python loss_compare.py --baseline <log_a> --candidate <log_b>
    python loss_compare.py --baseline <log_a> --candidate <log_b> --output <out_dir>
    python loss_compare.py --baseline <log_a> --candidate <log_b> --tolerance 1e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Import the log parser from the training-log-visualization skill
_VIZ_DIR = (
    Path(__file__).resolve().parents[2] / "training-log-visualization" / "scripts"
)
sys.path.insert(0, str(_VIZ_DIR))
from train_log_plot import read_training_metrics  # noqa: E402

METRIC_KEYS = ["loss", "grad_norm"]
METRIC_LABELS = {"loss": "Loss", "grad_norm": "Grad Norm"}
DEFAULT_TOLERANCE = 1e-5
NONFINITE_METRIC_RE = re.compile(
    r"\b(?:loss|grad_norm):\s*[-+]?(?:nan|inf)\b", re.IGNORECASE
)


@dataclass
class RunMetrics:
    metric: str
    steps: list[int]
    values: list[float]
    source: str


@dataclass
class CompareResult:
    metric: str
    baseline_metrics: RunMetrics
    candidate_metrics: RunMetrics
    common_steps: list[int]
    abs_diffs: list[float]
    rel_diffs: list[float]
    max_abs_diff: float
    max_abs_diff_step: int
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    nans_in_baseline: int
    nans_in_candidate: int
    infs_in_baseline: int
    infs_in_candidate: int


def extract_metric(records: list[dict], metric: str, source: str) -> RunMetrics | None:
    steps, values = [], []
    for rec in records:
        if metric in rec:
            steps.append(int(rec["step"]))
            values.append(float(rec[metric]))
    if not steps:
        print(f"WARNING: metric '{metric}' not found in {source}")
        return None
    return RunMetrics(metric=metric, steps=steps, values=values, source=source)


def compare_runs(baseline: RunMetrics, candidate: RunMetrics) -> CompareResult:
    b_map = dict(zip(baseline.steps, baseline.values))
    c_map = dict(zip(candidate.steps, candidate.values))
    common_steps = sorted(set(b_map) & set(c_map))
    if not common_steps:
        print(f"ERROR: No common steps for metric '{baseline.metric}'")
        sys.exit(1)

    abs_diffs, rel_diffs = [], []
    nans_b = nans_c = infs_b = infs_c = 0

    for step in common_steps:
        bv, cv = b_map[step], c_map[step]
        if np.isnan(bv):
            nans_b += 1
        if np.isnan(cv):
            nans_c += 1
        if np.isinf(bv):
            infs_b += 1
        if np.isinf(cv):
            infs_c += 1
        ad = abs(bv - cv)
        abs_diffs.append(ad)
        denom = max(abs(bv), abs(cv))
        rel_diffs.append(
            ad / denom if denom > 0 else (0.0 if ad == 0 else float("inf"))
        )

    return CompareResult(
        metric=baseline.metric,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        common_steps=common_steps,
        abs_diffs=abs_diffs,
        rel_diffs=rel_diffs,
        max_abs_diff=max(abs_diffs),
        max_abs_diff_step=common_steps[abs_diffs.index(max(abs_diffs))],
        mean_abs_diff=float(np.mean(abs_diffs)),
        max_rel_diff=max(rel_diffs),
        mean_rel_diff=float(np.mean(rel_diffs)),
        nans_in_baseline=nans_b,
        nans_in_candidate=nans_c,
        infs_in_baseline=infs_b,
        infs_in_candidate=infs_c,
    )


def find_nonfinite_metric_lines(log_path: str) -> list[int]:
    lines: list[int] = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if NONFINITE_METRIC_RE.search(line):
                lines.append(line_no)
    return lines


def first_exceedance(
    result: CompareResult, tolerance: float
) -> tuple[int | None, float | None, float | None]:
    for step, abs_diff, rel_diff in zip(
        result.common_steps, result.abs_diffs, result.rel_diffs
    ):
        if abs_diff > tolerance:
            return step, abs_diff, rel_diff
    return None, None, None


def plot_overlay(
    baseline: RunMetrics,
    candidate: RunMetrics,
    result: CompareResult,
    output_dir: str,
    filename: str,
    tolerance: float,
) -> str:
    label = METRIC_LABELS.get(result.metric, result.metric)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(7, 4.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax_top.plot(
        baseline.steps, baseline.values, "b-", lw=1.0, alpha=0.8, label="Baseline"
    )
    ax_top.plot(
        candidate.steps, candidate.values, "r--", lw=1.0, alpha=0.8, label="Candidate"
    )
    ax_top.set_ylabel(label)
    ax_top.legend(loc="upper right")
    ax_top.set_title(f"{label} — Baseline vs Candidate")
    ax_top.grid(True, alpha=0.3)

    ax_bottom.plot(result.common_steps, result.abs_diffs, "k-", lw=0.8)
    ax_bottom.set_ylabel("|diff|")
    ax_bottom.set_xlabel("Step")
    ax_bottom.set_yscale("log")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.axhline(
        y=tolerance, color="r", linestyle=":", alpha=0.5, label=f"{tolerance:.0e}"
    )
    ax_bottom.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def generate_csv(results: list[CompareResult], output_dir: str) -> str:
    out_path = os.path.join(output_dir, "diff_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "step", "baseline", "candidate", "abs_diff", "rel_diff"])
        for r in results:
            b_map = dict(zip(r.baseline_metrics.steps, r.baseline_metrics.values))
            c_map = dict(zip(r.candidate_metrics.steps, r.candidate_metrics.values))
            for i, step in enumerate(r.common_steps):
                w.writerow(
                    [
                        r.metric,
                        step,
                        b_map.get(step, ""),
                        c_map.get(step, ""),
                        r.abs_diffs[i],
                        r.rel_diffs[i],
                    ]
                )
    return out_path


def generate_json(
    results: list[CompareResult],
    baseline_path: str,
    candidate_path: str,
    output_dir: str,
    passed: bool,
    tolerance: float,
) -> str:
    out_path = os.path.join(output_dir, "report.json")
    report = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "baseline_log": baseline_path,
        "candidate_log": candidate_path,
        "tolerance": tolerance,
        "passed": passed,
        "metrics": [],
    }
    for r in results:
        first_step, first_abs_diff, first_rel_diff = first_exceedance(r, tolerance)
        report["metrics"].append(
            {
                "metric": r.metric,
                "num_common_steps": len(r.common_steps),
                "max_abs_diff": r.max_abs_diff,
                "max_abs_diff_step": r.max_abs_diff_step,
                "first_exceed_step": first_step,
                "first_exceed_abs_diff": first_abs_diff,
                "first_exceed_rel_diff": first_rel_diff,
                "mean_abs_diff": r.mean_abs_diff,
                "max_rel_diff": r.max_rel_diff,
                "mean_rel_diff": r.mean_rel_diff,
                "nans_in_baseline": r.nans_in_baseline,
                "nans_in_candidate": r.nans_in_candidate,
                "infs_in_baseline": r.infs_in_baseline,
                "infs_in_candidate": r.infs_in_candidate,
            }
        )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path


def check_pass(results: list[CompareResult], tolerance: float) -> bool:
    for r in results:
        if (
            r.nans_in_baseline != r.nans_in_candidate
            or r.infs_in_baseline != r.infs_in_candidate
        ):
            return False
        if r.max_abs_diff > tolerance:
            return False
    return True


def print_summary(results: list[CompareResult], tolerance: float, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  Numerical Stability Report — {status}")
    print(f"{'='*60}")
    print(f"  Tolerance: {tolerance:.1e}\n")
    for r in results:
        print(f"  [{r.metric}]")
        print(f"    Common steps:       {len(r.common_steps)}")
        print(f"    Max absolute diff:  {r.max_abs_diff:.6e}")
        print(f"    Mean absolute diff: {r.mean_abs_diff:.6e}")
        print(f"    Max relative diff:  {r.max_rel_diff:.6e}")
        print(f"    Mean relative diff: {r.mean_rel_diff:.6e}")
        if r.nans_in_baseline or r.nans_in_candidate:
            print(
                f"    NaN: baseline={r.nans_in_baseline}, candidate={r.nans_in_candidate}"
            )
        if r.infs_in_baseline or r.infs_in_candidate:
            print(
                f"    Inf: baseline={r.infs_in_baseline}, candidate={r.infs_in_candidate}"
            )
        print()
    if not passed:
        print("  FAILURE: Differences exceed tolerance.")
        for r in results:
            if r.max_abs_diff > tolerance:
                idx = r.abs_diffs.index(r.max_abs_diff)
                print(
                    f"    [{r.metric}] max diff at step {r.common_steps[idx]}: "
                    f"{r.max_abs_diff:.6e}"
                )
    else:
        print("  SUCCESS: All metrics within tolerance.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare loss/grad_norm between two training stdout logs"
    )
    parser.add_argument(
        "--baseline", required=True, help="Baseline training log file (.log)"
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate training log file (.log)"
    )
    parser.add_argument(
        "--output", default=os.getcwd(), help="Output directory for reports"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Max allowed absolute diff (default: {DEFAULT_TOLERANCE:.0e})",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=METRIC_KEYS,
        help=f"Metrics to compare (default: {' '.join(METRIC_KEYS)})",
    )
    args = parser.parse_args()

    for label, path in [("baseline", args.baseline), ("candidate", args.candidate)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} log file not found: {path}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    has_nonfinite_metrics = False
    for label, path in [("baseline", args.baseline), ("candidate", args.candidate)]:
        line_numbers = find_nonfinite_metric_lines(path)
        if line_numbers:
            shown = ", ".join(str(n) for n in line_numbers[:10])
            suffix = " ..." if len(line_numbers) > 10 else ""
            print(
                f"ERROR: {label} log contains NaN/Inf metric values at lines: "
                f"{shown}{suffix}"
            )
            has_nonfinite_metrics = True
    if has_nonfinite_metrics:
        sys.exit(1)

    print(f"Parsing baseline: {args.baseline}")
    baseline_records, bw = read_training_metrics(args.baseline)
    for w in bw:
        print(f"  [WARNING] {w}")
    print(f"  {len(baseline_records)} steps parsed")

    print(f"Parsing candidate: {args.candidate}")
    candidate_records, cw = read_training_metrics(args.candidate)
    for w in cw:
        print(f"  [WARNING] {w}")
    print(f"  {len(candidate_records)} steps parsed")

    if bw or cw:
        print("ERROR: Parser warnings found; refusing numerical comparison.")
        sys.exit(1)

    results: list[CompareResult] = []
    for metric in args.metrics:
        bl = extract_metric(baseline_records, metric, args.baseline)
        cd = extract_metric(candidate_records, metric, args.candidate)
        if bl is None or cd is None:
            continue
        result = compare_runs(bl, cd)
        results.append(result)
        fname = f"{metric}_comparison.png"
        plot_path = plot_overlay(bl, cd, result, args.output, fname, args.tolerance)
        print(f"  Plot: {plot_path}")

    if not results:
        print("ERROR: No metrics could be compared.")
        sys.exit(1)

    print(f"CSV:  {generate_csv(results, args.output)}")
    passed = check_pass(results, args.tolerance)
    print(
        f"JSON: {generate_json(results, args.baseline, args.candidate, args.output, passed, args.tolerance)}"
    )
    print_summary(results, args.tolerance, passed)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
