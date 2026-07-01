# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serializes a WorkloadGraph (and everything it recursively contains) to
JSON. `dataclasses.asdict()` recursively converts every nested dataclass
(including dataclasses stored as dict/list values, e.g. `StepGraph.nodes:
dict[str, OpNode]`) into plain dicts/lists, which `json.dumps` can then
serialize directly (tuples become JSON arrays)."""

from __future__ import annotations

import dataclasses
import json

from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph


def workload_graph_to_dict(workload_graph: WorkloadGraph) -> dict:
    return dataclasses.asdict(workload_graph)


def export_json(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workload_graph_to_dict(workload_graph), f, indent=2, ensure_ascii=False)
