# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 OpNode: the smallest modeling unit in the four-layer simulator IR.

See spec: https://github.com/ZhanluModelSim/workload-model-platform/blob/master/spec/L0-OpNode.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


@dataclass
class OpNode:
    """A single normalized operator invocation captured during a train step."""

    op_id: int
    op_type: str
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    attrs: dict[str, Any]
    predecessors: list[str]
    successors: list[str]
    flops: int = 0
    peak_mem: int = 0
    param_mem: int = 0
    comm_bytes: int = 0
    annotations: dict[str, Any] = field(default_factory=dict)
    seq_idx: int = 0

    def export_detail_csv(self, path: str) -> None:
        """Export this single op's full details as a one-row CSV."""
        import csv
        ann = self.annotations
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "op_id", "seq_idx", "op_type", "raw_op_type", "phase",
                "execution_kind", "is_recompute",
                "inputs_shape", "outputs_shape", "inputs_dtype", "outputs_dtype",
                "flops", "peak_mem", "param_mem", "comm_bytes",
                "repeat_count", "module_path", "comm_dim", "comm_ranks",
                "predecessors", "successors",
            ])
            shapes = lambda metas: ";".join("[" + ",".join(str(d) for d in m.shape) + "]" for m in metas)
            dtypes = lambda metas: ";".join(m.dtype for m in metas)
            w.writerow([
                self.op_id, self.seq_idx, self.op_type,
                ann.get("raw_op_type", ""), ann.get("phase", ""),
                ann.get("execution_kind", ""), ann.get("is_recompute", False),
                shapes(self.inputs), shapes(self.outputs),
                dtypes(self.inputs), dtypes(self.outputs),
                self.flops, self.peak_mem, self.param_mem, self.comm_bytes,
                ann.get("repeat_count", 1), ann.get("module_path", ""),
                ann.get("comm_dim", ""), ann.get("comm_ranks", ""),
                ";".join(str(p) for p in self.predecessors),
                ";".join(str(s) for s in self.successors),
            ])
