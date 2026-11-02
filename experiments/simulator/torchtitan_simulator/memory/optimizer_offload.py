# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Project optimizer state out of device memory, retaining its transfer inventory."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from torchtitan_simulator.memory.estimator import _build_timeline, _model_peak_active_bytes, _phase_peak_active_bytes

if TYPE_CHECKING:
    from torchtitan_simulator.memory.records import MemoryPlan


def apply_optimizer_offload_budget(memory: MemoryPlan) -> MemoryPlan:
    """Ignore transient state buffers by policy; captured lifetimes remain untouched."""
    if memory.optimizer_state_bytes is None:
        raise ValueError("Optimizer offload requires an observed optimizer state inventory")
    if memory.offload_optimizer_states:
        raise ValueError("Optimizer offload budget has already been applied")
    result = replace(memory, offload_optimizer_states=True, notes=[*memory.notes])
    phase_at = {event.seq_idx: event.phase for event in memory.raw_events}
    timeline = _build_timeline(result.device_budget_lifetimes(), phase_at)
    peaks = _phase_peak_active_bytes(timeline, memory.raw_events)
    peak = max(timeline, key=lambda item: item.active_bytes_after, default=None)
    result.timeline_events = timeline
    result.peak_active_bytes = peak.active_bytes_after if peak else 0
    result.peak_seq_idx = peak.seq_idx if peak else 0
    result.peak_phase = peak.phase if peak else ""
    result.model_active_bytes_peak = _model_peak_active_bytes(timeline, memory.raw_events)
    result.forward_peak_active_bytes = peaks["forward"]
    result.backward_peak_active_bytes = peaks["backward"]
    result.optimizer_peak_active_bytes = peaks["optimizer"]
    # Optimizer state is the only steady-state projection. Once offloaded,
    # the remaining observed lifetimes and activation buffers define all peaks.
    result.steady_state_forward_peak_active_bytes = peaks["forward"]
    result.steady_state_backward_peak_active_bytes = peaks["backward"]
    result.steady_state_optimizer_peak_active_bytes = peaks["optimizer"]
    result.steady_state_peak_active_bytes = max(peaks.values())
    result.notes.append(
        "Optimizer states are host-resident in F/B/O; aggregate full-duplex transfers overlap "
        "device optimizer updates. Transient per-tensor state buffers are ignored by policy; "
        "optimizer_state_bytes remains the observed per-direction payload inventory."
    )
    return result
