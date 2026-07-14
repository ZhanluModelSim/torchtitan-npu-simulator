# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP full-parameter residency model.

FSDP all-gather can materialize a transient full parameter before compute and
release it after the consuming layer. In meta simulation this residency is not
always represented as an independent tensor id: some paths expose only the
local shard/parameter object. This plugin fills that specific gap while leaving
normal tensor liveness to the generic estimator.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_npu.simulator.memory.records import RawMemoryEvent, TensorLifetime, TensorRef


def _comm_field(comm: Any, name: str, default: Any = "") -> Any:
    return getattr(comm, name, default) if comm is not None else default


def _is_fsdp_allgather(event: RawMemoryEvent, comm: Any) -> bool:
    primitive = _comm_field(comm, "comm_primitive", "") or event.raw_op_type.removeprefix("comm.")
    comm_dim = str(_comm_field(comm, "comm_dim", ""))
    return primitive == "allgather" and "fsdp" in comm_dim.lower()


def _full_param_bytes(event: RawMemoryEvent, comm: Any) -> int:
    if event.outputs:
        return sum(ref.num_bytes for ref in event.outputs)
    volume_bytes = int(_comm_field(comm, "volume_bytes", 0) or 0)
    world_size = int(_comm_field(comm, "world_size", 1) or 1)
    return volume_bytes * world_size


def _shape_dtype_from_event(event: RawMemoryEvent) -> tuple[tuple[int, ...], str]:
    refs: tuple[TensorRef, ...] = event.outputs or event.inputs
    if not refs:
        return (), ""
    if event.outputs:
        if len(event.outputs) == 1:
            return event.outputs[0].shape, event.outputs[0].dtype
        return (), event.outputs[0].dtype
    ref = refs[0]
    return ref.shape, ref.dtype


class FSDPFullParamResidencyPlugin(MemoryModelPlugin):
    """Ensure FSDP all-gather full-param residency is represented once."""

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        if context.fsdp_residency_events:
            return self._apply_explicit_markers(context)

        return self._apply_comm_fallback(context)

    def _apply_explicit_markers(self, context: MemoryModelContext) -> list[TensorLifetime]:
        marked_tensor_ids = {
            tensor_id
            for event in context.fsdp_residency_events
            for tensor_id in event.tensor_ids
        }
        removed = 0
        for tensor_id in marked_tensor_ids:
            if context.lifetimes_by_tensor_id.pop(tensor_id, None) is not None:
                removed += 1

        alloc_seqs = sorted(
            event.seq_idx for event in context.fsdp_residency_events if event.action == "alloc"
        )
        shortened_staging_buffers = 0
        for event in context.events:
            comm = context.comm_by_op.get(event.op_id)
            if not _is_fsdp_l2_allgather(event, comm):
                continue
            alloc_idx = bisect_left(alloc_seqs, event.seq_idx)
            if alloc_idx >= len(alloc_seqs):
                continue
            release_seq = alloc_seqs[alloc_idx]
            for ref in event.outputs:
                lifetime = context.lifetimes_by_tensor_id.get(ref.tensor_id)
                if lifetime is None or lifetime.birth_seq != event.seq_idx:
                    continue
                lifetime.kind = "comm_buffer"
                lifetime.reason = "fsdp_allgather_staging"
                lifetime.death_seq = min(lifetime.death_seq, release_seq)
                shortened_staging_buffers += 1

        active_by_group: dict[str, TensorLifetime] = {}
        synthesized: list[TensorLifetime] = []
        unmatched_frees = 0
        for event in sorted(context.fsdp_residency_events, key=lambda item: item.seq_idx):
            if event.action == "alloc":
                lifetime = TensorLifetime(
                    tensor_id=f"fsdp_full_param:{event.group_id}:{event.seq_idx}",
                    kind="fsdp_full_param",
                    num_bytes=event.num_bytes,
                    birth_seq=event.seq_idx,
                    death_seq=event.seq_idx,
                    producer_op=-1,
                    producer_raw_op="fsdp.unshard",
                    producer_phase=event.phase,
                    reason="fsdp_explicit_residency",
                )
                active_by_group[event.group_id] = lifetime
                synthesized.append(lifetime)
            elif event.action == "free":
                lifetime = active_by_group.pop(event.group_id, None)
                if lifetime is None:
                    unmatched_frees += 1
                    continue
                lifetime.death_seq = max(lifetime.birth_seq, event.seq_idx)
                lifetime.consumer_phases.append(event.phase)

        context.notes.append(
            "FSDP residency plugin used explicit unshard/reshard markers: "
            f"{len(synthesized)} residency intervals, {removed} generic tensor lifetimes replaced, "
            f"{shortened_staging_buffers} all-gather staging lifetimes shortened."
        )
        if active_by_group or unmatched_frees:
            context.notes.append(
                "FSDP residency marker imbalance: "
                f"{len(active_by_group)} allocs without free, {unmatched_frees} frees without alloc."
            )
        return synthesized

    def _apply_comm_fallback(self, context: MemoryModelContext) -> list[TensorLifetime]:
        event_by_op = {event.op_id: event for event in context.events}
        synthesized: list[TensorLifetime] = []

        for event in context.events:
            comm = context.comm_by_op.get(event.op_id)
            if not _is_fsdp_allgather(event, comm):
                continue

            visible_lifetimes = [
                context.lifetimes_by_tensor_id[ref.tensor_id]
                for ref in event.outputs
                if ref.tensor_id in context.lifetimes_by_tensor_id
            ]
            if visible_lifetimes:
                for lifetime in visible_lifetimes:
                    lifetime.kind = "fsdp_full_param"
                    lifetime.reason = "fsdp_allgather"
                continue

            num_bytes = _full_param_bytes(event, comm)
            if num_bytes <= 0:
                continue

            shape, dtype = _shape_dtype_from_event(event)
            lifetime = TensorLifetime(
                tensor_id=f"fsdp_full_param:{event.op_id}",
                kind="fsdp_full_param",
                num_bytes=num_bytes,
                birth_seq=event.seq_idx,
                death_seq=event.seq_idx,
                producer_op=event.op_id,
                producer_raw_op=event.raw_op_type,
                producer_phase=event.phase,
                shape=shape,
                dtype=dtype,
                reason="fsdp_allgather_residency",
            )

            dst_entry_op = int(_comm_field(comm, "dst_entry_op", 0) or 0)
            consumer_event = event_by_op.get(dst_entry_op)
            if consumer_event is not None and consumer_event.seq_idx >= event.seq_idx:
                lifetime.mark_consumer(consumer_event.op_id, consumer_event.seq_idx, consumer_event.phase)

            synthesized.append(lifetime)

        if synthesized:
            context.notes.append(
                f"FSDP residency plugin synthesized {len(synthesized)} full-param lifetimes "
                "when meta capture did not expose independent tensors."
            )
        return synthesized


def _is_fsdp_l2_allgather(event: RawMemoryEvent, comm: Any) -> bool:
    primitive = _comm_field(comm, "comm_primitive", "") or event.raw_op_type.removeprefix("comm.")
    return primitive == "allgather" and _comm_field(comm, "comm_layer", "") == "L2"
