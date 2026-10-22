# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run native-size DSV4 Meta captures in isolated processes and verify exports."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PRESETS = ["deepseek_v4_smoketest"] + [
    f"deepseek_v4_{flavor}_baseline_{precision}"
    for flavor in ("flash", "pro", "pro_20t")
    for precision in ("bf16", "mxfp8")
]
GEOMETRY = {"smoketest": (4, 256), "flash": (43, 4096), "pro": (61, 7168), "pro_20t": (96, 12288)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(args):
    from collections import Counter
    from dataclasses import asdict

    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import export_result, simulate
    from torchtitan_simulator.validation import validate_workload

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=False)
    small = args.case == "deepseek_v4_smoketest"
    options = SimulationConfig(
        config=args.case,
        world_size=16,
        logical_rank=15,
        data_parallel_replicate_degree=2,
        expert_parallel_degree=2,
        tensor_parallel_degree=2,
        context_parallel_degree=2,
        num_mtp_modules=1,
        seq_len=512,
        local_batch_size=1,
        activation_checkpoint_mode=args.mode,
        csv_max_ranks=1,
        output_dir=str(destination),
        output_formats=["text", "html", "dot", "json", "csv", "mem"] if small else ["text", "csv", "mem"],
    )
    started = time.monotonic()
    workload = simulate(options)
    report = validate_workload(workload)
    annotations = workload.iteration.schedule.annotations
    geometry = annotations["model_configuration"]
    flavor = "smoketest" if small else args.case.removeprefix("deepseek_v4_").split("_baseline_")[0]
    require((geometry["num_layers"], geometry["dim"]) == GEOMETRY[flavor], "native geometry changed")
    require(geometry["num_mtp_layers"] == 1, "MTP not exercised")
    quantization = annotations["quantization_configuration"]
    require(quantization["enabled"] == args.case.endswith("_mxfp8"), "wrong native recipe selection")
    memory = annotations["memory_plan"]
    kinds = Counter(event.execution_kind for event in memory.raw_events)
    if args.mode != "none":
        require(kinds["recompute"] > 0, "AC did not execute recomputation")
    export_result(workload, options)
    summary = json.loads((destination / "memory/memory_summary.json").read_text())
    require(summary["active_bytes_peak"] == report["peak_active_bytes"], "memory consumer peak differs")
    nodes = {str(node.op_id): node for template in workload.step_templates.values() for node in template.nodes.values()}
    with (destination / "kernel_summary/rank_0.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {key for key, node in nodes.items() if not node.annotations.get("metadata_view")}
    require({row["op_id"] for row in rows} == expected and len(rows) == len(expected), "kernel CSV lost/duplicated ops")
    require(len(list((destination / "kernel_summary").glob("rank_*.csv"))) == 1, "CSV rank limit ineffective")
    with (destination / "schedule_plan.csv").open() as stream:
        reader = csv.DictReader(stream)
        actions = []
        for row in reader:
            if row["seq_idx"] == "# DataSlots":
                break
            actions.append(row)
    require(
        {row["action_id"] for row in actions} == set(workload.schedule_plan.action_map),
        "schedule CSV differs from plan",
    )
    if small:
        exported = json.loads((destination / "simulation_result.json").read_text())
        require(set(exported["step_templates"]) == set(workload.step_templates), "JSON template references differ")
        require(
            exported["schedule_plan"]["actions"]
            == json.loads(json.dumps([asdict(action) for action in workload.schedule_plan.actions])),
            "JSON schedule differs from authoritative plan",
        )
        for suffix in ("txt", "html", "dot"):
            text = (destination / f"simulation_result.{suffix}").read_text()
            require(bool(text.strip()), f"empty {suffix} output")
    report.update(
        status="passed",
        options=asdict(options),
        model=geometry,
        capture_coordinates=annotations["capture_coordinates"],
        batch_configuration=annotations["batch_configuration"],
        quantized_parameters=len(quantization["quantized_parameters"]),
        execution_kinds=dict(kinds),
        raw_kernel_counts=dict(Counter(event.raw_op_type for event in memory.raw_events)),
        unknown_cost_counts=annotations["cost_unknown_counts"],
        parameter_layouts=annotations["parameter_layouts"],
        elapsed_seconds=time.monotonic() - started,
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        packages={
            name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "torchao", "torchtitan-simulator")
        },
        exports={
            str(path.relative_to(destination)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in destination.rglob("*")
            if path.is_file()
        },
    )
    (destination / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "graph_nodes", "gradient_parameters", "peak_active_bytes", "elapsed_seconds")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--presets", choices=PRESETS, nargs="+", default=PRESETS)
    parser.add_argument(
        "--modes", choices=["none", "full", "selective"], nargs="+", default=["none", "full", "selective"]
    )
    parser.add_argument("--case", choices=PRESETS)
    parser.add_argument("--mode", choices=["none", "full", "selective"])
    args = parser.parse_args()
    if args.case:
        require(args.mode is not None, "--case requires --mode")
        run_case(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "scope": "Meta simulation acceptance", "cases": []}
    failed = False
    for preset in args.presets:
        for mode in args.modes:
            label = f"{preset}-{mode}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                preset,
                "--mode",
                mode,
                "--output-dir",
                str(args.output_dir / label),
            ]
            with (args.output_dir / f"{label}.log").open("w") as log:
                process = subprocess.run(
                    command,
                    env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            passed = process.returncode == 0
            failed |= not passed
            manifest["cases"].append(
                {
                    "label": label,
                    "command": command,
                    "status": "passed" if passed else "failed",
                    "exit_code": process.returncode,
                }
            )
            (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"{label}: {'passed' if passed else 'FAILED'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
