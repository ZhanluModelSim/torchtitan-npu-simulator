# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static active-tensor-bytes estimator.

This is intentionally not an allocator simulator. It scans a completed
meta capture, assigns lifetimes from producer/consumer order, and reports
active tensor bytes that are explainable from static shapes.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from torchtitan_simulator.capture.tensor_utils import (
    dtype_to_str,
    normalize_supported_dtype,
    tensor_volume_bytes,
)
from torchtitan_simulator.memory.activation_checkpoint import ActivationCheckpointPlugin
from torchtitan_simulator.memory.activation_offload import ActivationOffloadPlugin
from torchtitan_simulator.memory.alias_rules import is_alias_event, is_mutation_event
from torchtitan_simulator.memory.fsdp_residency import FSDPFullParamResidencyPlugin
from torchtitan_simulator.memory.plugins import (
    MemoryModelContext,
    MissingParameterGradient,
    ParameterTensorMetadata,
)
from torchtitan_simulator.memory.records import (
    AutogradSavedTensorEvent,
    CheckpointBoundaryEvent,
    FSDPResidencyEvent,
    MemoryPlan,
    MemoryTimelineEvent,
    RawMemoryEvent,
    StorageLifetimeRecord,
    TensorLifetime,
    TensorRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def _to_local_tensor(value: object) -> torch.Tensor | None:
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            local_tensor = getattr(value, "_local_tensor", None)
            if isinstance(local_tensor, torch.Tensor):
                return local_tensor
            return value.to_local()
    except Exception:
        pass
    if isinstance(value, torch.Tensor):
        return value
    return None


def _snapshot_parameters(
    model_parts: Iterable[nn.Module],
    parameter_storage_dtype: str | None = None,
    *,
    counted_device_type: str = "",
) -> tuple[list[TensorLifetime], set[int], list[MissingParameterGradient], list[ParameterTensorMetadata]]:
    lifetimes: list[TensorLifetime] = []
    seen_params: set[int] = set()
    param_ids: set[int] = set()
    missing_gradients: list[MissingParameterGradient] = []
    parameter_tensors: list[ParameterTensorMetadata] = []
    for part_idx, model in enumerate(model_parts):
        for name, param in model.named_parameters(recurse=True):
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            tensor = _to_local_tensor(param)
            if tensor is None or (counted_device_type and tensor.device.type != counted_device_type):
                continue
            tid = id(tensor)
            param_ids.add(tid)
            dtype = dtype_to_str(tensor.dtype)
            shape = tuple(int(d) for d in tensor.shape)
            num_bytes = tensor_volume_bytes(shape, dtype)
            modeled_num_bytes = (
                tensor_volume_bytes(shape, parameter_storage_dtype) if parameter_storage_dtype is not None else None
            )
            lifetimes.append(
                TensorLifetime(
                    tensor_id=f"param:{part_idx}:{name}",
                    kind="parameter_shard",
                    num_bytes=num_bytes,
                    birth_seq=-1,
                    death_seq=-1,
                    producer_op=-1,
                    producer_raw_op="model_parameter",
                    producer_phase="init",
                    shape=shape,
                    dtype=dtype,
                    modeled_num_bytes=modeled_num_bytes,
                    modeled_dtype=parameter_storage_dtype or "",
                    residency_policy=(
                        "dtype_override"
                        if parameter_storage_dtype is not None and parameter_storage_dtype != dtype
                        else "resident"
                    ),
                    reason="persistent_param",
                    storage_key=str(tensor.untyped_storage()._cdata),
                )
            )
            parameter_tensors.append(
                ParameterTensorMetadata(
                    name=f"{part_idx}:{name}",
                    num_bytes=num_bytes,
                    shape=shape,
                    dtype=dtype,
                    storage_key=str(tensor.untyped_storage()._cdata),
                )
            )
            if param.requires_grad and getattr(param, "grad", None) is None:
                missing_gradients.append(
                    MissingParameterGradient(
                        name=f"{part_idx}:{name}",
                        num_bytes=num_bytes,
                        shape=shape,
                        dtype=dtype,
                    )
                )
    return lifetimes, param_ids, missing_gradients, parameter_tensors


def _is_persistent_storage_alias(ref: TensorRef, persistent_storage_keys: set[str]) -> bool:
    """Only a witnessed persistent storage may be excluded from activation bytes.

    Model state remains alive while the plan is built, so its storage handles
    cannot be reused. Capture's generation suffix distinguishes other dead
    storage handles without conflating same-shaped values or FSDP buffers.
    """
    storage_key = ref.storage_key.partition(":")[0]
    return bool(storage_key) and storage_key in persistent_storage_keys


def _snapshot_model_buffers(model_parts, parameter_storage_keys, counted_device_type):
    lifetimes = []
    ids = set()
    storage_keys = set(parameter_storage_keys)
    for part_idx, model in enumerate(model_parts):
        for name, buffer in model.named_buffers():
            tensor = _to_local_tensor(buffer)
            if tensor is None or (counted_device_type and tensor.device.type != counted_device_type):
                continue
            ids.add(id(tensor))
            storage = tensor.untyped_storage()
            key = str(storage._cdata)
            if key in storage_keys:
                continue
            storage_keys.add(key)
            lifetimes.append(
                TensorLifetime(
                    tensor_id=f"buffer:{part_idx}:{name}",
                    kind="persistent_buffer",
                    num_bytes=storage.nbytes(),
                    birth_seq=-1,
                    death_seq=-1,
                    producer_op=-1,
                    producer_raw_op="model_buffer",
                    producer_phase="init",
                    shape=tuple(tensor.shape),
                    dtype=dtype_to_str(tensor.dtype),
                    reason="registered_model_buffer",
                    storage_key=key,
                )
            )
    return lifetimes, ids, storage_keys


def _external_lifetime(ref: TensorRef, event: RawMemoryEvent) -> TensorLifetime:
    return TensorLifetime(
        tensor_id=f"external:{ref.tensor_id}",
        kind="external_input",
        num_bytes=ref.storage_bytes if ref.storage_key and ref.storage_bytes else ref.num_bytes,
        birth_seq=event.seq_idx,
        death_seq=event.seq_idx,
        producer_op=-1,
        producer_raw_op="external_input",
        producer_phase=event.phase,
        shape=ref.shape,
        dtype=ref.dtype,
        reason="first_observed_as_input",
        storage_key=ref.storage_key,
    )


def _output_lifetime(ref: TensorRef, event: RawMemoryEvent, kind: str, reason: str) -> TensorLifetime:
    return TensorLifetime(
        tensor_id=f"tensor:{ref.tensor_id}",
        kind=kind,
        num_bytes=ref.storage_bytes if ref.storage_key and ref.storage_bytes else ref.num_bytes,
        birth_seq=event.seq_idx,
        death_seq=event.seq_idx,
        producer_op=event.op_id,
        producer_raw_op=event.raw_op_type,
        producer_phase=event.phase,
        shape=ref.shape,
        dtype=ref.dtype,
        reason=reason,
        storage_key=ref.storage_key,
    )


def _alias_lifetime(ref: TensorRef, event: RawMemoryEvent, base_id: int) -> TensorLifetime:
    return TensorLifetime(
        tensor_id=f"alias:{ref.tensor_id}",
        kind="alias",
        num_bytes=0,
        birth_seq=event.seq_idx,
        death_seq=event.seq_idx,
        producer_op=event.op_id,
        producer_raw_op=event.raw_op_type,
        producer_phase=event.phase,
        alias_of=f"tensor:{base_id}",
        shape=ref.shape,
        dtype=ref.dtype,
        reason="alias_rule",
        storage_key=ref.storage_key,
    )


def _comm_event_by_op(comm_events: Iterable[Any] | None) -> dict[int, Any]:
    result: dict[int, Any] = {}
    if comm_events is None:
        return result
    for event in comm_events:
        op_id = int(getattr(event, "op_id", 0) or 0)
        if op_id:
            result[op_id] = event
    return result


def _resolve_alias(tensor_id: int, alias_base_by_tensor_id: dict[int, int]) -> int:
    seen: set[int] = set()
    current = tensor_id
    while current in alias_base_by_tensor_id and current not in seen:
        seen.add(current)
        current = alias_base_by_tensor_id[current]
    return current


def _classify_output(event: RawMemoryEvent, comm_by_op: dict[int, Any]) -> tuple[str, str]:
    if event.op_id in comm_by_op or event.raw_op_type.startswith("comm."):
        comm = comm_by_op.get(event.op_id)
        primitive = getattr(comm, "comm_primitive", "") if comm is not None else event.raw_op_type.removeprefix("comm.")
        comm_dim = getattr(comm, "comm_dim", "") if comm is not None else ""
        if primitive == "allgather" and "fsdp" in comm_dim.lower():
            return "fsdp_full_param", "fsdp_allgather"
        return "comm_buffer", f"comm_{primitive}"
    return "op_output", "producer"


def _finalize_kind(lifetime: TensorLifetime) -> None:
    if lifetime.kind in {
        "parameter_shard",
        "optimizer_state",
        "persistent_buffer",
        "external_input",
        "alias",
        "fsdp_full_param",
        "checkpoint_recompute_temp",
        "checkpoint_saved_activation",
        "checkpoint_saved_for_recompute",
        "offloaded_activation",
    }:
        return
    if lifetime.producer_phase == "backward" and "optimizer" in lifetime.consumer_phases:
        lifetime.kind = "gradient_accumulator"
        lifetime.reason = "backward_to_optimizer"
        return
    if lifetime.kind == "comm_buffer":
        return
    if not lifetime.consumer_ops:
        lifetime.kind = "dead_temp_output"
        lifetime.reason = "no_consumer"
    elif lifetime.producer_phase == "forward" and "backward" in lifetime.consumer_phases:
        lifetime.kind = "activation"
        if lifetime.reason not in {"autograd_saved_tensor", "backward_use_without_saved_slot"}:
            lifetime.reason = "forward_to_backward"
    else:
        lifetime.kind = "temporary"
        lifetime.reason = "last_consumer"


def _build_timeline(
    lifetimes: list[TensorLifetime],
    phase_by_seq: dict[int, str],
) -> list[MemoryTimelineEvent]:
    edges: list[tuple[int, int, TensorLifetime, str]] = []
    for lifetime in lifetimes:
        if lifetime.resident_num_bytes <= 0:
            continue
        edges.append((lifetime.birth_seq, 0, lifetime, "alloc"))
        edges.append((lifetime.death_seq, 1, lifetime, "free"))
    edges.sort(key=lambda item: (item[0], item[1], item[2].tensor_id))

    active = 0
    timeline: list[MemoryTimelineEvent] = []
    phase_positions = sorted(phase_by_seq)
    for seq_idx, _order, lifetime, action in edges:
        if action == "alloc":
            active += lifetime.resident_num_bytes
            phase = lifetime.producer_phase
            op_id = lifetime.producer_op
        else:
            active -= lifetime.resident_num_bytes
            # Activation checkpointing can shorten a forward lifetime even
            # though its original use-def chain also has a backward
            # recomputation consumer. The actual release sequence is the
            # authoritative phase for trace spans and phase peaks.
            phase = phase_by_seq.get(
                seq_idx,
                lifetime.consumer_phases[-1] if lifetime.consumer_phases else lifetime.producer_phase,
            )
            op_id = lifetime.consumer_ops[-1] if lifetime.consumer_ops else lifetime.producer_op
        position = bisect_right(phase_positions, seq_idx) - 1
        if position >= 0:
            phase = phase_by_seq[phase_positions[position]]
        timeline.append(
            MemoryTimelineEvent(
                seq_idx=seq_idx,
                phase=phase,
                op_id=op_id,
                action=action,
                tensor_id=lifetime.tensor_id,
                kind=lifetime.kind,
                num_bytes=lifetime.resident_num_bytes,
                active_bytes_after=active,
                reason=lifetime.reason,
            )
        )
    return timeline


def _phase_peak_active_bytes(
    timeline: list[MemoryTimelineEvent],
    events: list[RawMemoryEvent],
) -> dict[str, int]:
    """Return phase peaks, including memory already live on phase entry."""
    peaks = dict.fromkeys(("forward", "backward", "optimizer"), 0)
    phase_at = {event.seq_idx: event.phase for event in events}
    memory_at: dict[int, list[MemoryTimelineEvent]] = defaultdict(list)
    for event in timeline:
        memory_at[event.seq_idx].append(event)
        phase_at.setdefault(event.seq_idx, event.phase)
    active = 0
    phase = "init"
    for seq in sorted(phase_at.keys() | memory_at.keys()):
        phase = phase_at.get(seq, phase)
        if phase in peaks:
            peaks[phase] = max(peaks[phase], active)
        for event in memory_at.get(seq, ()):
            active = event.active_bytes_after
            if phase in peaks:
                peaks[phase] = max(peaks[phase], active)
    return peaks


def _model_peak_active_bytes(
    timeline: list[MemoryTimelineEvent],
    events: list[RawMemoryEvent],
) -> int:
    """Return the rank-local peak before optimizer execution begins."""
    optimizer_start = min(
        (event.seq_idx for event in events if event.phase == "optimizer"),
        default=None,
    )
    return max(
        (event.active_bytes_after for event in timeline if optimizer_start is None or event.seq_idx < optimizer_start),
        default=0,
    )


def estimate_static_memory(
    raw_events: Iterable[RawMemoryEvent],
    *,
    model_parts: Iterable[nn.Module] | None = None,
    comm_events: Iterable[Any] | None = None,
    fsdp_residency_events: Iterable[FSDPResidencyEvent] | None = None,
    checkpoint_boundary_events: Iterable[CheckpointBoundaryEvent] | None = None,
    autograd_saved_tensor_events: Iterable[AutogradSavedTensorEvent] | None = None,
    storage_lifetimes: Iterable[StorageLifetimeRecord] | None = None,
    initial_tensors: Iterable[TensorRef] | None = None,
    optimizer_state_tensors: Iterable[TensorRef] | None = None,
    counted_device_type: str = "",
    parameter_storage_dtype: str | None = None,
    offload_ac_saved_tensors: bool = False,
) -> MemoryPlan:
    events = sorted(raw_events, key=lambda event: event.seq_idx)
    parameter_storage_dtype = normalize_supported_dtype(parameter_storage_dtype) if parameter_storage_dtype else None
    model_parts = list(model_parts or [])
    param_lifetimes, param_ids, missing_parameter_gradients, parameter_tensors = _snapshot_parameters(
        model_parts,
        parameter_storage_dtype,
        counted_device_type=counted_device_type,
    )
    parameter_storage_keys = {parameter.storage_key for parameter in parameter_tensors}
    buffer_lifetimes, buffer_ids, persistent_storage_keys = _snapshot_model_buffers(
        model_parts,
        parameter_storage_keys,
        counted_device_type,
    )
    end_seq = max((event.seq_idx for event in events), default=0) + 1
    for lifetime in param_lifetimes:
        lifetime.death_seq = end_seq

    comm_by_op = _comm_event_by_op(comm_events)
    lifetimes_by_tensor_id: dict[int, TensorLifetime] = {}
    alias_base_by_tensor_id: dict[int, int] = {}
    alias_lifetimes: list[TensorLifetime] = []
    parameter_materialization_ids: set[int] = set()
    storage_roots: dict[str, int] = {}
    unclassified_ops: list[dict[str, Any]] = []
    notes = [
        "P0 estimates active tensor bytes from static tensor metadata; "
        "it does not model allocator reserved/cache or kernel workspace.",
        "Allocation identity uses captured storage provenance, with op-name rules for legacy events.",
    ]

    def storage_root(ref: TensorRef) -> int:
        root = _resolve_alias(ref.tensor_id, alias_base_by_tensor_id)
        if ref.storage_key and ref.storage_key in storage_roots:
            root = _resolve_alias(storage_roots[ref.storage_key], alias_base_by_tensor_id)
        elif ref.alias_of is not None:
            root = _resolve_alias(ref.alias_of, alias_base_by_tensor_id)
        if root != ref.tensor_id:
            alias_base_by_tensor_id[ref.tensor_id] = root
        if ref.storage_key:
            storage_roots[ref.storage_key] = root
        return root

    initial_tensors = list(initial_tensors or [])
    for ref in initial_tensors:
        if counted_device_type and ref.device.split(":")[0] != counted_device_type:
            continue
        root = storage_root(ref)
        if root in param_ids or root in buffer_ids or _is_persistent_storage_alias(ref, persistent_storage_keys):
            continue
        if root in lifetimes_by_tensor_id:
            continue
        lifetimes_by_tensor_id[root] = TensorLifetime(
            tensor_id=f"external:{root}",
            kind="external_input",
            num_bytes=ref.storage_bytes if ref.storage_key and ref.storage_bytes else ref.num_bytes,
            birth_seq=-1,
            death_seq=end_seq,
            producer_op=-1,
            producer_raw_op="capture_input",
            producer_phase="init",
            shape=ref.shape,
            dtype=ref.dtype,
            storage_key=ref.storage_key,
            reason="preallocated_capture_input",
        )

    for event in events:
        for ref in event.inputs:
            if counted_device_type and ref.device.split(":")[0] != counted_device_type:
                continue
            root_tensor_id = storage_root(ref)
            if (
                root_tensor_id in param_ids
                or root_tensor_id in buffer_ids
                or root_tensor_id in parameter_materialization_ids
            ):
                continue
            lifetime = lifetimes_by_tensor_id.get(root_tensor_id)
            if lifetime is None:
                if _is_persistent_storage_alias(ref, persistent_storage_keys):
                    parameter_materialization_ids.add(root_tensor_id)
                    continue
                lifetime = _external_lifetime(ref, event)
                lifetime.tensor_id = f"external:{root_tensor_id}"
                lifetimes_by_tensor_id[root_tensor_id] = lifetime
            lifetime.mark_consumer(event.op_id, event.seq_idx, event.phase)

        input_ids = {ref.tensor_id for ref in event.inputs}
        alias = is_alias_event(event)
        mutation = is_mutation_event(event)
        for ref in event.outputs:
            if counted_device_type and ref.device.split(":")[0] != counted_device_type:
                continue
            root_tensor_id = storage_root(ref)
            if (
                root_tensor_id in param_ids
                or root_tensor_id in buffer_ids
                or root_tensor_id in parameter_materialization_ids
                or _is_persistent_storage_alias(ref, persistent_storage_keys)
            ):
                parameter_materialization_ids.add(root_tensor_id)
                continue
            if ref.storage_key and root_tensor_id in lifetimes_by_tensor_id:
                # A returned alias or write does not allocate its storage again.
                lifetime = lifetimes_by_tensor_id[root_tensor_id]
                lifetime.mark_consumer(event.op_id, event.seq_idx, event.phase)
                if ref.tensor_id != root_tensor_id:
                    alias_lifetimes.append(_alias_lifetime(ref, event, root_tensor_id))
                continue
            if mutation and ref.tensor_id in input_ids:
                continue
            if alias and event.inputs and not ref.storage_key:
                base_tensor_id = _resolve_alias(event.inputs[0].tensor_id, alias_base_by_tensor_id)
                alias_base_by_tensor_id[ref.tensor_id] = base_tensor_id
                alias_lifetimes.append(_alias_lifetime(ref, event, base_tensor_id))
                continue
            kind, reason = _classify_output(event, comm_by_op)
            lifetime = _output_lifetime(ref, event, kind, reason)
            lifetime.tensor_id = f"tensor:{root_tensor_id}"
            lifetimes_by_tensor_id[root_tensor_id] = lifetime

        if event.op_type == "unknown" and event.outputs:
            unclassified_ops.append(
                {
                    "seq_idx": event.seq_idx,
                    "op_id": event.op_id,
                    "raw_op_type": event.raw_op_type,
                    "phase": event.phase,
                    "output_bytes": sum(ref.num_bytes for ref in event.outputs),
                }
            )

    unknown_inputs = [item for item in lifetimes_by_tensor_id.values() if item.kind == "external_input"]
    if unknown_inputs:
        notes.append(
            f"Capture observed {len(unknown_inputs)} external inputs without persistent-parameter storage provenance "
            f"({sum(item.num_bytes for item in unknown_inputs)} logical bytes); kept in the memory inventory."
        )

    plugin_context = MemoryModelContext(
        events=events,
        comm_by_op=comm_by_op,
        lifetimes_by_tensor_id=lifetimes_by_tensor_id,
        param_ids=param_ids,
        fsdp_residency_events=list(fsdp_residency_events or []),
        missing_parameter_gradients=missing_parameter_gradients,
        parameter_tensors=parameter_tensors,
        checkpoint_boundary_events=list(checkpoint_boundary_events or []),
        autograd_saved_tensors=list(autograd_saved_tensor_events or []),
        alias_base_by_tensor_id=alias_base_by_tensor_id,
        storage_roots=storage_roots,
        offload_ac_saved_tensors=offload_ac_saved_tensors,
        notes=notes,
    )
    plugin_lifetimes = FSDPFullParamResidencyPlugin().apply(plugin_context)
    ActivationCheckpointPlugin().apply(plugin_context)
    from torchtitan_simulator.memory.saved_retention import retain_saved_storages

    retain_saved_storages(plugin_context)
    ActivationOffloadPlugin().apply(plugin_context)
    observed_storages = list(storage_lifetimes or [])
    capture_end = max(
        [
            end_seq,
            *(item.release_seq for item in observed_storages),
            *(item.release_seq for item in plugin_context.autograd_saved_tensors),
        ],
    )
    for observed in observed_storages:
        root = storage_roots.get(observed.storage_key)
        if root is None:
            continue
        lifetime = lifetimes_by_tensor_id.get(root)
        if lifetime is None:
            continue
        death = observed.release_seq if observed.release_seq >= 0 else capture_end
        if death < max([lifetime.birth_seq, *lifetime.consumer_seqs]):
            raise ValueError("Observed storage release precedes a captured producer/consumer")
        lifetime.death_seq = death
    for lifetime in (*param_lifetimes, *buffer_lifetimes):
        lifetime.death_seq = capture_end
    if observed_storages:
        notes.append(
            "Storage ownership was observed through weak storage references at dispatcher boundaries; "
            "live Python/C++ owners remain resident through capture end. Explicit FSDP free markers "
            "override storage-object ownership because reshard can resize live storage to zero."
        )
    if plugin_context.missing_parameter_gradients:
        notes.append("Missing parameter gradients are unobserved; no synthetic gradient storage was added.")

    lifetimes = [
        *param_lifetimes,
        *buffer_lifetimes,
        *lifetimes_by_tensor_id.values(),
        *alias_lifetimes,
        *plugin_lifetimes,
    ]
    state_keys = None
    if optimizer_state_tensors is not None:
        state_keys = set()
        for ref in optimizer_state_tensors:
            if counted_device_type and ref.device.split(":")[0] != counted_device_type:
                continue
            if _is_persistent_storage_alias(ref, persistent_storage_keys):
                continue
            if not ref.storage_key:
                raise ValueError("Optimizer state lacks captured storage provenance")
            state_keys.add(ref.storage_key)
        witnessed = set()
        for lifetime in lifetimes:
            if lifetime.storage_key in state_keys and not lifetime.alias_of:
                lifetime.kind = "optimizer_state"
                lifetime.reason = "optimizer_owned_state"
                witnessed.add(lifetime.storage_key)
        if witnessed != state_keys:
            raise ValueError("Optimizer state storage is absent from the captured memory inventory")
    for lifetime in lifetimes:
        _finalize_kind(lifetime)

    phase_by_seq = {event.seq_idx: event.phase for event in plugin_context.fsdp_residency_events}
    phase_by_seq.update({event.seq_idx: event.phase for event in events})
    timeline = _build_timeline(lifetimes, phase_by_seq)
    peak_event = max(timeline, key=lambda item: item.active_bytes_after, default=None)
    phase_peaks = _phase_peak_active_bytes(timeline, events)
    plan = MemoryPlan(
        counted_device_type=counted_device_type,
        parameter_storage_dtype=parameter_storage_dtype or "",
        offload_ac_saved_tensors=offload_ac_saved_tensors,
        persistent_param_bytes=sum(item.resident_num_bytes for item in param_lifetimes),
        persistent_buffer_bytes=sum(item.resident_num_bytes for item in buffer_lifetimes),
        peak_active_bytes=peak_event.active_bytes_after if peak_event else 0,
        model_active_bytes_peak=_model_peak_active_bytes(timeline, events),
        forward_peak_active_bytes=phase_peaks["forward"],
        backward_peak_active_bytes=phase_peaks["backward"],
        optimizer_peak_active_bytes=phase_peaks["optimizer"],
        peak_seq_idx=peak_event.seq_idx if peak_event else 0,
        peak_phase=peak_event.phase if peak_event else "",
        raw_events=events,
        tensor_lifetimes=sorted(lifetimes, key=lambda item: (item.birth_seq, item.tensor_id)),
        checkpoint_tensors=plugin_context.checkpoint_tensors,
        activation_offload_tensors=plugin_context.activation_offload_tensors,
        autograd_saved_tensors=plugin_context.autograd_saved_tensors,
        fsdp_residency_events=plugin_context.fsdp_residency_events,
        storage_lifetimes=observed_storages,
        initial_tensors=initial_tensors,
        timeline_events=timeline,
        unclassified_ops=unclassified_ops,
        notes=notes,
    )
    if state_keys is not None:
        states = [lifetime for lifetime in lifetimes if lifetime.kind == "optimizer_state"]
        steady_lifetimes = [
            replace(lifetime, birth_seq=-1, death_seq=capture_end) if lifetime.kind == "optimizer_state" else lifetime
            for lifetime in lifetimes
        ]
        steady_timeline = _build_timeline(steady_lifetimes, phase_by_seq)
        steady_peaks = _phase_peak_active_bytes(steady_timeline, events)
        plan.optimizer_state_bytes = sum(lifetime.resident_num_bytes for lifetime in states)
        plan.optimizer_state_storage_count = len(states)
        plan.steady_state_peak_active_bytes = max((event.active_bytes_after for event in steady_timeline), default=0)
        plan.steady_state_forward_peak_active_bytes = steady_peaks["forward"]
        plan.steady_state_backward_peak_active_bytes = steady_peaks["backward"]
        plan.steady_state_optimizer_peak_active_bytes = steady_peaks["optimizer"]
        notes.append(
            "Steady-state peaks project actual optimizer-owned device storages as resident from capture start; "
            "other captured lifetimes are unchanged. This is not a second executed iteration or GA replay."
        )
    else:
        notes.append("Steady-state peaks unavailable: optimizer callable exposes no public state owner.")
    return plan
