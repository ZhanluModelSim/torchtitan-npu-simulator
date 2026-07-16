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

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.tensor_utils import dtype_to_str, tensor_volume_bytes
from torchtitan_npu.simulator.memory.activation_checkpoint import ActivationCheckpointPlugin
from torchtitan_npu.simulator.memory.alias_rules import is_alias_event, is_mutation_event
from torchtitan_npu.simulator.memory.fsdp_residency import FSDPFullParamResidencyPlugin
from torchtitan_npu.simulator.memory.gradient_residency import MissingParameterGradientPlugin
from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MissingParameterGradient
from torchtitan_npu.simulator.memory.records import (
    FSDPResidencyEvent,
    MemoryPlan,
    MemoryTimelineEvent,
    RawMemoryEvent,
    TensorLifetime,
    TensorRef,
)


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
) -> tuple[list[TensorLifetime], set[int], list[MissingParameterGradient]]:
    lifetimes: list[TensorLifetime] = []
    seen_params: set[int] = set()
    param_ids: set[int] = set()
    missing_gradients: list[MissingParameterGradient] = []
    for part_idx, model in enumerate(model_parts):
        for name, param in model.named_parameters(recurse=True):
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            tensor = _to_local_tensor(param)
            if tensor is None:
                continue
            tid = id(tensor)
            param_ids.add(tid)
            dtype = dtype_to_str(tensor.dtype)
            shape = tuple(int(d) for d in tensor.shape)
            lifetimes.append(
                TensorLifetime(
                    tensor_id=f"param:{part_idx}:{name}",
                    kind="parameter_shard",
                    num_bytes=tensor_volume_bytes(shape, dtype),
                    birth_seq=-1,
                    death_seq=-1,
                    producer_op=-1,
                    producer_raw_op="model_parameter",
                    producer_phase="init",
                    shape=shape,
                    dtype=dtype,
                    reason="persistent_param",
                )
            )
            if param.requires_grad and getattr(param, "grad", None) is None:
                missing_gradients.append(
                    MissingParameterGradient(
                        name=f"{part_idx}:{name}",
                        num_bytes=tensor_volume_bytes(shape, dtype),
                        shape=shape,
                        dtype=dtype,
                    )
                )
    return lifetimes, param_ids, missing_gradients


def _external_lifetime(ref: TensorRef, event: RawMemoryEvent) -> TensorLifetime:
    return TensorLifetime(
        tensor_id=f"external:{ref.tensor_id}",
        kind="external_input",
        num_bytes=ref.num_bytes,
        birth_seq=event.seq_idx,
        death_seq=event.seq_idx,
        producer_op=-1,
        producer_raw_op="external_input",
        producer_phase=event.phase,
        shape=ref.shape,
        dtype=ref.dtype,
        reason="first_observed_as_input",
    )


def _output_lifetime(ref: TensorRef, event: RawMemoryEvent, kind: str, reason: str) -> TensorLifetime:
    return TensorLifetime(
        tensor_id=f"tensor:{ref.tensor_id}",
        kind=kind,
        num_bytes=ref.num_bytes,
        birth_seq=event.seq_idx,
        death_seq=event.seq_idx,
        producer_op=event.op_id,
        producer_raw_op=event.raw_op_type,
        producer_phase=event.phase,
        shape=ref.shape,
        dtype=ref.dtype,
        reason=reason,
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
        "external_input",
        "alias",
        "fsdp_full_param",
        "checkpoint_recompute_temp",
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
        lifetime.reason = "forward_to_backward"
    else:
        lifetime.kind = "temporary"
        lifetime.reason = "last_consumer"


def _build_timeline(lifetimes: list[TensorLifetime]) -> list[MemoryTimelineEvent]:
    edges: list[tuple[int, int, TensorLifetime, str]] = []
    for lifetime in lifetimes:
        if lifetime.num_bytes <= 0:
            continue
        edges.append((lifetime.birth_seq, 0, lifetime, "alloc"))
        edges.append((lifetime.death_seq, 1, lifetime, "free"))
    edges.sort(key=lambda item: (item[0], item[1], item[2].tensor_id))

    active = 0
    timeline: list[MemoryTimelineEvent] = []
    for seq_idx, _order, lifetime, action in edges:
        if action == "alloc":
            active += lifetime.num_bytes
            phase = lifetime.producer_phase
            op_id = lifetime.producer_op
        else:
            active -= lifetime.num_bytes
            phase = lifetime.consumer_phases[-1] if lifetime.consumer_phases else lifetime.producer_phase
            op_id = lifetime.consumer_ops[-1] if lifetime.consumer_ops else lifetime.producer_op
        timeline.append(
            MemoryTimelineEvent(
                seq_idx=seq_idx,
                phase=phase,
                op_id=op_id,
                action=action,
                tensor_id=lifetime.tensor_id,
                kind=lifetime.kind,
                num_bytes=lifetime.num_bytes,
                active_bytes_after=active,
                reason=lifetime.reason,
            )
        )
    return timeline


def estimate_static_memory(
    raw_events: Iterable[RawMemoryEvent],
    *,
    model_parts: Iterable[nn.Module] | None = None,
    comm_events: Iterable[Any] | None = None,
    fsdp_residency_events: Iterable[FSDPResidencyEvent] | None = None,
) -> MemoryPlan:
    events = sorted(raw_events, key=lambda event: event.seq_idx)
    param_lifetimes, param_ids, missing_parameter_gradients = _snapshot_parameters(model_parts or [])
    end_seq = max((event.seq_idx for event in events), default=0) + 1
    for lifetime in param_lifetimes:
        lifetime.death_seq = end_seq

    comm_by_op = _comm_event_by_op(comm_events)
    lifetimes_by_tensor_id: dict[int, TensorLifetime] = {}
    alias_base_by_tensor_id: dict[int, int] = {}
    alias_lifetimes: list[TensorLifetime] = []
    unclassified_ops: list[dict[str, Any]] = []
    notes = [
        "P0 estimates active tensor bytes from static tensor metadata; it does not model allocator reserved/cache or kernel workspace.",
        "Alias and mutation handling uses conservative op-name rules.",
    ]

    for event in events:
        for ref in event.inputs:
            root_tensor_id = _resolve_alias(ref.tensor_id, alias_base_by_tensor_id)
            if root_tensor_id in param_ids:
                continue
            lifetime = lifetimes_by_tensor_id.get(root_tensor_id)
            if lifetime is None:
                lifetime = _external_lifetime(ref, event)
                lifetimes_by_tensor_id[root_tensor_id] = lifetime
            lifetime.mark_consumer(event.op_id, event.seq_idx, event.phase)

        input_ids = {ref.tensor_id for ref in event.inputs}
        alias = is_alias_event(event)
        mutation = is_mutation_event(event)
        for ref in event.outputs:
            if ref.tensor_id in param_ids:
                continue
            if mutation and ref.tensor_id in input_ids:
                continue
            if alias and event.inputs:
                base_tensor_id = _resolve_alias(event.inputs[0].tensor_id, alias_base_by_tensor_id)
                alias_base_by_tensor_id[ref.tensor_id] = base_tensor_id
                alias_lifetimes.append(_alias_lifetime(ref, event, base_tensor_id))
                continue
            kind, reason = _classify_output(event, comm_by_op)
            lifetimes_by_tensor_id[ref.tensor_id] = _output_lifetime(ref, event, kind, reason)

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

    plugin_context = MemoryModelContext(
        events=events,
        comm_by_op=comm_by_op,
        lifetimes_by_tensor_id=lifetimes_by_tensor_id,
        param_ids=param_ids,
        fsdp_residency_events=list(fsdp_residency_events or []),
        missing_parameter_gradients=missing_parameter_gradients,
        notes=notes,
    )
    plugin_lifetimes = FSDPFullParamResidencyPlugin().apply(plugin_context)
    ActivationCheckpointPlugin().apply(plugin_context)
    plugin_lifetimes.extend(MissingParameterGradientPlugin().apply(plugin_context))

    lifetimes = [*param_lifetimes, *lifetimes_by_tensor_id.values(), *alias_lifetimes, *plugin_lifetimes]
    for lifetime in lifetimes:
        _finalize_kind(lifetime)

    timeline = _build_timeline(lifetimes)
    peak_event = max(timeline, key=lambda item: item.active_bytes_after, default=None)
    plan = MemoryPlan(
        persistent_param_bytes=sum(item.num_bytes for item in param_lifetimes),
        peak_active_bytes=peak_event.active_bytes_after if peak_event else 0,
        peak_seq_idx=peak_event.seq_idx if peak_event else 0,
        peak_phase=peak_event.phase if peak_event else "",
        raw_events=events,
        tensor_lifetimes=sorted(lifetimes, key=lambda item: (item.birth_seq, item.tensor_id)),
        timeline_events=timeline,
        unclassified_ops=unclassified_ops,
        notes=notes,
    )
    return plan
