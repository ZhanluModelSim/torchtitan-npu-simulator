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
                f"FSDP residency plugin synthesized {len(synthesized)} full-param lifetimes when meta capture did not expose independent tensors."
            )
        return synthesized
