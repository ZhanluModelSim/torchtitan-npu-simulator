# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extension points for static memory modeling.

The core estimator owns generic use-def liveness. Plugins add narrow,
framework-specific residency models without mixing those policies into the
main event scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent, RawMemoryEvent, TensorLifetime


@dataclass(slots=True)
class MemoryModelContext:
    events: list[RawMemoryEvent]
    comm_by_op: dict[int, Any]
    lifetimes_by_tensor_id: dict[int, TensorLifetime]
    param_ids: set[int]
    fsdp_residency_events: list[FSDPResidencyEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class MemoryModelPlugin(Protocol):
    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        """Return extra lifetimes synthesized by this model."""
