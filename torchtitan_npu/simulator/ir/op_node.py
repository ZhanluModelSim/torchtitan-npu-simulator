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
