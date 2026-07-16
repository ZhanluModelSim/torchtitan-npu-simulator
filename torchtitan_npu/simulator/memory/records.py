# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataclasses shared by capture, static memory planning, and exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorRef:
    tensor_id: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    num_bytes: int


@dataclass(frozen=True, slots=True)
class RawMemoryEvent:
    event_id: int
    op_id: int
    seq_idx: int
    raw_op_type: str
    op_type: str
    phase: str
    module_path: str
    inputs: tuple[TensorRef, ...]
    outputs: tuple[TensorRef, ...]
    pp_stage: int = -1
    pp_mb_idx: int = -1


@dataclass(frozen=True, slots=True)
class FSDPResidencyEvent:
    group_id: str
    action: str
    seq_idx: int
    phase: str
    num_bytes: int
    tensor_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class TensorLifetime:
    tensor_id: str
    kind: str
    num_bytes: int
    birth_seq: int
    death_seq: int
    producer_op: int
    producer_raw_op: str = ""
    producer_phase: str = ""
    consumer_ops: list[int] = field(default_factory=list)
    consumer_seqs: list[int] = field(default_factory=list)
    consumer_phases: list[str] = field(default_factory=list)
    alias_of: str = ""
    shape: tuple[int, ...] = ()
    dtype: str = ""
    reason: str = ""

    def mark_consumer(self, op_id: int, seq_idx: int, phase: str) -> None:
        self.consumer_ops.append(op_id)
        self.consumer_seqs.append(seq_idx)
        self.consumer_phases.append(phase)
        if seq_idx > self.death_seq:
            self.death_seq = seq_idx


@dataclass(frozen=True, slots=True)
class MemoryTimelineEvent:
    seq_idx: int
    phase: str
    op_id: int
    action: str
    tensor_id: str
    kind: str
    num_bytes: int
    active_bytes_after: int
    reason: str = ""


@dataclass(slots=True)
class MemoryPlan:
    metric: str = "active_tensor_bytes"
    persistent_param_bytes: int = 0
    peak_active_bytes: int = 0
    forward_peak_active_bytes: int = 0
    backward_peak_active_bytes: int = 0
    optimizer_peak_active_bytes: int = 0
    peak_seq_idx: int = 0
    peak_phase: str = ""
    raw_events: list[RawMemoryEvent] = field(default_factory=list)
    tensor_lifetimes: list[TensorLifetime] = field(default_factory=list)
    timeline_events: list[MemoryTimelineEvent] = field(default_factory=list)
    unclassified_ops: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "persistent_param_bytes": self.persistent_param_bytes,
            "active_bytes_peak": self.peak_active_bytes,
            "forward_active_bytes_peak": self.forward_peak_active_bytes,
            "backward_active_bytes_peak": self.backward_peak_active_bytes,
            "optimizer_active_bytes_peak": self.optimizer_peak_active_bytes,
            "peak_seq_idx": self.peak_seq_idx,
            "peak_phase": self.peak_phase,
            "raw_memory_event_count": len(self.raw_events),
            "tensor_lifetime_count": len(self.tensor_lifetimes),
            "timeline_event_count": len(self.timeline_events),
            "unclassified_op_count": len(self.unclassified_ops),
            "included": [
                "local parameter tensors",
                "external inputs and labels observed by dispatch",
                "non-alias operator outputs by use-def liveness",
                "collective communication outputs observed by CommEvent",
            ],
            "excluded": [
                "allocator reserved/cache",
                "fragmentation",
                "kernel workspace",
                "device internal temporary buffers",
            ],
            "notes": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.to_summary_dict()
        data["tensor_lifetimes"] = [asdict(item) for item in self.tensor_lifetimes]
        data["timeline_events"] = [asdict(item) for item in self.timeline_events]
        data["unclassified_ops"] = list(self.unclassified_ops)
        return data
