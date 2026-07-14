# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSV/JSON exports for static memory plans."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from torchtitan_npu.simulator.memory.records import MemoryPlan


def export_memory_plan(plan: MemoryPlan, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "memory_summary.json"), "w", encoding="utf-8") as f:
        json.dump(plan.to_summary_dict(), f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "memory_events.csv"), "w", newline="", encoding="utf-8") as f:
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

    with open(os.path.join(out_dir, "tensor_lifetimes.csv"), "w", newline="", encoding="utf-8") as f:
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
        with open(os.path.join(out_dir, "unclassified_memory_ops.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = ["seq_idx", "op_id", "raw_op_type", "phase", "output_bytes"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in plan.unclassified_ops:
                writer.writerow(item)
