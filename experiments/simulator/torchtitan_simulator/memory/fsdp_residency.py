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

from torchtitan_simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_simulator.memory.records import RawMemoryEvent, TensorLifetime, TensorRef


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
        original_lifetimes = dict(context.lifetimes_by_tensor_id)
        marked_tensor_ids = {
            context.alias_base_by_tensor_id.get(tensor_id, tensor_id)
            for event in context.fsdp_residency_events
            for tensor_id in event.tensor_ids
        }
        marked_storage_keys = {key for event in context.fsdp_residency_events for key in event.storage_keys}
        marked_tensor_ids.update(
            context.storage_roots[key] for key in marked_storage_keys if key in context.storage_roots
        )
        removed = 0
        for tensor_id in marked_tensor_ids:
            if context.lifetimes_by_tensor_id.pop(tensor_id, None) is not None:
                removed += 1

        # ``DTensor.to_local()`` can expose an unsharded parameter through a
        # fresh tensor identity. When no dispatch-visible alias connects it to
        # the FSDP tracked tensor, generic use-def liveness sees it as an
        # external input and retains it until backward. Restrict this recovery
        # to explicit FSDP marker mode: the marker below remains the source of
        # truth for the full-parameter residency interval and byte count.
        removed_unsharded_aliases = 0
        removed_unsharded_alias_bytes = 0
        # Shape recovery is only for legacy captures without storage keys.
        if not marked_storage_keys:
            for tensor_id, lifetime in list(context.lifetimes_by_tensor_id.items()):
                if not _is_unsharded_parameter_alias(lifetime, context):
                    continue
                context.lifetimes_by_tensor_id.pop(tensor_id)
                removed_unsharded_aliases += 1
                removed_unsharded_alias_bytes += lifetime.num_bytes

        alloc_seqs = sorted(event.seq_idx for event in context.fsdp_residency_events if event.action == "alloc")
        alloc_by_transition = {
            marker.transition_id: marker.seq_idx
            for marker in context.fsdp_residency_events
            if marker.action == "alloc" and marker.transition_id
        }
        shortened_staging_buffers = 0
        for event in context.events:
            comm = context.comm_by_op.get(event.op_id)
            if not _is_fsdp_l2_allgather(event, comm):
                continue
            alloc_idx = bisect_left(alloc_seqs, event.seq_idx)
            if alloc_idx >= len(alloc_seqs):
                continue
            transition = str(_comm_field(comm, "fsdp_transition_id", ""))
            if transition and transition not in alloc_by_transition:
                # A mistargeted prefetch is released by finalize_backward,
                # rather than the next unrelated group's unshard.
                continue
            release_seq = alloc_by_transition[transition] if transition else alloc_seqs[alloc_idx]
            for ref in event.outputs:
                root = context.tensor_root(ref.tensor_id, ref.storage_key)
                lifetime = context.lifetimes_by_tensor_id.get(root)
                if lifetime is None or lifetime.birth_seq > event.seq_idx:
                    continue
                lifetime.kind = "comm_buffer"
                lifetime.reason = "fsdp_allgather_staging"
                lifetime.death_seq = release_seq if transition else min(lifetime.death_seq, release_seq)
                shortened_staging_buffers += 1

        active_by_group: dict[str, TensorLifetime] = {}
        synthesized: list[TensorLifetime] = []
        launch_by_transition = {
            str(_comm_field(context.comm_by_op.get(event.op_id), "fsdp_transition_id", "")): event
            for event in context.events
            if _is_fsdp_l2_allgather(event, context.comm_by_op.get(event.op_id))
        }
        for release in context.fsdp_residency_events:
            if release.action != "staging_free":
                continue
            launch = launch_by_transition.get(release.transition_id)
            if launch is None or launch.seq_idx > release.seq_idx:
                raise ValueError("FSDP staging release lacks a matching captured all-gather launch")
            roots = {context.tensor_root(ref.tensor_id, ref.storage_key) for ref in launch.outputs}
            visible = [original_lifetimes[root] for root in roots if root in original_lifetimes]
            for root in roots:
                context.lifetimes_by_tensor_id.pop(root, None)
            birth_seq = min([launch.seq_idx] + [item.birth_seq for item in visible])
            synthesized.append(
                TensorLifetime(
                    tensor_id=f"fsdp_pending_staging:{release.transition_id}",
                    kind="comm_buffer",
                    num_bytes=release.num_bytes,
                    birth_seq=birth_seq,
                    death_seq=release.seq_idx,
                    producer_op=launch.op_id,
                    producer_raw_op=launch.raw_op_type,
                    producer_phase=launch.phase,
                    reason=(
                        "fsdp_allgather_observed_release"
                        if release.transition_id in alloc_by_transition
                        else "fsdp_prefetch_finalize_release"
                    ),
                    storage_key=(release.storage_keys[0] if len(release.storage_keys) == 1 else ""),
                )
            )
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

        if active_by_group:
            capture_end = max(
                [event.seq_idx for event in context.events] + [event.seq_idx for event in context.fsdp_residency_events]
            )
            for lifetime in active_by_group.values():
                lifetime.death_seq = max(lifetime.birth_seq, capture_end)
                lifetime.reason = "fsdp_explicit_residency_open_at_capture_end"

        context.notes.append(
            "FSDP residency plugin used explicit unshard/reshard markers: "
            f"{len(synthesized)} residency intervals, {removed} generic tensor lifetimes replaced, "
            f"{shortened_staging_buffers} all-gather staging lifetimes shortened, "
            f"{removed_unsharded_aliases} unsharded parameter aliases removed "
            f"({removed_unsharded_alias_bytes} bytes)."
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


def _is_unsharded_parameter_alias(lifetime: TensorLifetime, context: MemoryModelContext) -> bool:
    """Whether an external tensor is an expanded local FSDP parameter shard."""
    if lifetime.kind != "external_input" or lifetime.producer_raw_op != "external_input":
        return False
    for parameter in context.parameter_tensors:
        if len(lifetime.shape) != len(parameter.shape) or not lifetime.shape:
            continue
        if lifetime.shape[1:] != parameter.shape[1:]:
            continue
        local_leading_dim = parameter.shape[0]
        if local_leading_dim <= 0 or lifetime.shape[0] < local_leading_dim:
            continue
        if lifetime.shape[0] % local_leading_dim == 0:
            return True
    return False
