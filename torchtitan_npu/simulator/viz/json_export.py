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

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    import json

    _HAS_ORJSON = False

from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph


def workload_graph_to_dict(workload_graph: WorkloadGraph) -> dict:
    return dataclasses.asdict(workload_graph)


def export_json(workload_graph: WorkloadGraph, path: str) -> None:
    data = workload_graph_to_dict(workload_graph)
    if _HAS_ORJSON:
        with open(path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    else:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
