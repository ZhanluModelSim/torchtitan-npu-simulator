# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static storage budget for the activation transfer policy, without DES replay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

from torchtitan_simulator.memory.estimator import _build_timeline, _model_peak_active_bytes, _phase_peak_active_bytes
from torchtitan_simulator.memory.records import TensorLifetime

if TYPE_CHECKING:
    from torchtitan_simulator.capture.activation_transfers import ActivationTransferPlan
    from torchtitan_simulator.memory.records import MemoryPlan


def apply_activation_transfer_budget(memory: MemoryPlan, plans: list[ActivationTransferPlan]) -> MemoryPlan:
    """Add deduplicated resident intervals, preserving raw lifetimes and O state."""
    # Alias records have zero bytes and may be born during backward. They must
    # never overwrite the allocation lifetime used to budget a storage.
    lifetimes = {item.storage_key or item.tensor_id: item for item in memory.tensor_lifetimes if item.num_bytes > 0}
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    sizes: dict[str, int] = {}
    for plan in plans:
        for storage, size in plan.storage_bytes:
            lifetime = lifetimes[storage]
            if lifetime.resident_num_bytes:
                continue
            sizes[storage] = max(sizes.get(storage, 0), size)
            forward_end = plan.forward_release_seq
            # Shared inputs remain live until their last native forward reader.
            forward_end = max(
                [
                    forward_end,
                    *(
                        seq
                        for seq, phase in zip(lifetime.consumer_seqs, lifetime.consumer_phases, strict=True)
                        if phase == "forward"
                    ),
                ]
            )
            spans[storage].append((lifetime.birth_seq, forward_end))
            backward_start = plan.backward_start_seq
            backward_end = max(
                [
                    plan.backward.end_seq,
                    *(
                        seq
                        for seq, phase in zip(lifetime.consumer_seqs, lifetime.consumer_phases, strict=True)
                        if phase == "backward"
                    ),
                ]
            )
            spans[storage].append((backward_start, backward_end))
    additions = []
    for storage, intervals in spans.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if start > end:
                raise ValueError(f"Invalid activation interval for {storage}: {start} > {end}; {lifetimes[storage]}")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for index, (start, end) in enumerate(merged):
            additions.append(
                TensorLifetime(
                    tensor_id=f"activation_buffer:{storage}:{index}",
                    kind="activation_transfer_buffer",
                    num_bytes=sizes[storage],
                    birth_seq=start,
                    death_seq=end,
                    producer_op=0,
                    residency_policy="neighbor_transfer_static",
                    storage_key=storage,
                    reason="Static activation transfer buffer; not a DES memory replay",
                )
            )
    events = memory.raw_events
    phase_at = {event.seq_idx: event.phase for event in events}
    phase_at.update({plan.backward_start_seq: "backward" for plan in plans})
    all_lifetimes = replace(memory, activation_buffer_lifetimes=additions).device_budget_lifetimes()
    timeline = _build_timeline(all_lifetimes, phase_at)
    peaks = _phase_peak_active_bytes(timeline, events)
    peak = max(timeline, key=lambda item: item.active_bytes_after, default=None)
    result = replace(
        memory,
        activation_buffer_lifetimes=additions,
        timeline_events=timeline,
        notes=[
            *memory.notes,
            "Activation copy buffers use storage-union residency across neighboring execution blocks; "
            "shared storage remains resident through native readers. This is a static policy budget, not DES replay.",
        ],
        peak_active_bytes=peak.active_bytes_after if peak else 0,
        peak_seq_idx=peak.seq_idx if peak else 0,
        peak_phase=peak.phase if peak else "",
        model_active_bytes_peak=_model_peak_active_bytes(timeline, events),
        forward_peak_active_bytes=peaks["forward"],
        backward_peak_active_bytes=peaks["backward"],
    )
    if any(plan.ready_seq > plan.forward.end_seq for plan in plans):
        result.notes.append(
            "late_payload_phase_drain: a layer's saved payload is produced after its forward boundary; "
            "the whole layer copy waits for actual producers and drains at F end. Buffers remain resident "
            "to F end, so more than two layers may overlap; per-ready payload splitting is not modeled."
        )
    if memory.optimizer_state_bytes is not None:
        capture_end = max((item.death_seq for item in all_lifetimes), default=0)
        steady = [
            replace(item, birth_seq=-1, death_seq=capture_end) if item.kind == "optimizer_state" else item
            for item in all_lifetimes
        ]
        steady_peaks = _phase_peak_active_bytes(_build_timeline(steady, phase_at), events)
        result.steady_state_forward_peak_active_bytes = steady_peaks["forward"]
        result.steady_state_backward_peak_active_bytes = steady_peaks["backward"]
        result.steady_state_peak_active_bytes = max(
            steady_peaks["forward"], steady_peaks["backward"], memory.steady_state_optimizer_peak_active_bytes or 0
        )
    return result
