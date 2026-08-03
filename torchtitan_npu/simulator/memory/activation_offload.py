# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified saved-activation offload modeling."""

from __future__ import annotations

import re

from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_npu.simulator.memory.records import (
    CheckpointTensorRecord,
    RawMemoryEvent,
    TensorLifetime,
    TensorRef,
)

_LAYER_PATH_PATTERN = re.compile(r"(?:^|\.)(layers\.\d+)(?:\.|$)")


def _layer_scope(module_path: str) -> str:
    match = _LAYER_PATH_PATTERN.search(module_path)
    return match.group(1) if match else ""


def _resolve_alias(tensor_id: int, alias_base_by_tensor_id: dict[int, int]) -> int:
    seen: set[int] = set()
    current = tensor_id
    while current in alias_base_by_tensor_id and current not in seen:
        seen.add(current)
        current = alias_base_by_tensor_id[current]
    return current


def _producer_ref(
    tensor_id: int,
    producer_event: RawMemoryEvent,
    alias_base_by_tensor_id: dict[int, int],
) -> TensorRef | None:
    for ref in producer_event.outputs:
        if _resolve_alias(ref.tensor_id, alias_base_by_tensor_id) == tensor_id:
            return ref
    return None


def _layer_occurrences(
    events: list[RawMemoryEvent],
    *,
    phase: str,
) -> dict[int, tuple[str, int]]:
    """Map event seq to a stable layer invocation marker and anchor seq."""
    occurrences: dict[int, tuple[str, int]] = {}
    current_key: tuple[str, int, int] | None = None
    anchor_seq = -1
    for event in events:
        if event.phase != phase:
            continue
        scope = _layer_scope(event.module_path)
        if not scope:
            continue
        key = (scope, event.pp_stage, event.pp_mb_idx)
        if key != current_key:
            current_key = key
            anchor_seq = event.seq_idx
        occurrences[event.seq_idx] = (scope, anchor_seq)
    return occurrences


def _normal_backward_consumers(
    lifetime: TensorLifetime,
    event_by_seq: dict[int, RawMemoryEvent],
) -> list[RawMemoryEvent]:
    return [
        event_by_seq[seq_idx]
        for seq_idx, phase in zip(
            lifetime.consumer_seqs,
            lifetime.consumer_phases,
            strict=True,
        )
        if phase == "backward"
        and seq_idx in event_by_seq
        and event_by_seq[seq_idx].execution_kind == "backward"
    ]


class ActivationOffloadPlugin(MemoryModelPlugin):
    """Offload every real forward activation retained for normal backward."""

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        if not context.offload_ac_saved_tensors:
            return []

        event_by_seq = {event.seq_idx: event for event in context.events}
        backward_occurrences = _layer_occurrences(context.events, phase="backward")
        forward_occurrences = _layer_occurrences(context.events, phase="forward")
        checkpoint_recorded_ids = {
            item.tensor_id
            for item in context.checkpoint_tensors
            if item.is_saved_activation
        }

        offloaded_count = 0
        offloaded_bytes = 0
        unattributed_count = 0
        unattributed_bytes = 0

        for tensor_id, lifetime in context.lifetimes_by_tensor_id.items():
            if (
                lifetime.producer_phase != "forward"
                or "backward" not in lifetime.consumer_phases
                or lifetime.kind in {
                    "external_input",
                    "fsdp_full_param",
                    "checkpoint_recompute_temp",
                }
            ):
                continue

            producer_event = event_by_seq.get(lifetime.birth_seq)
            if producer_event is None:
                continue

            lifetime.modeled_num_bytes = 0
            lifetime.residency_policy = "offloaded"
            if lifetime.kind not in {
                "checkpoint_saved_activation",
                "checkpoint_saved_for_recompute",
            }:
                lifetime.kind = "offloaded_activation"
                lifetime.reason = "forward_tensor_saved_for_backward_offloaded"
            offloaded_count += 1
            offloaded_bytes += lifetime.num_bytes

            if lifetime.tensor_id in checkpoint_recorded_ids:
                continue

            ref = _producer_ref(
                tensor_id,
                producer_event,
                context.alias_base_by_tensor_id,
            )
            backward_consumers = _normal_backward_consumers(lifetime, event_by_seq)
            targets: dict[tuple[str, int, int, int], RawMemoryEvent] = {}
            for event in backward_consumers:
                occurrence = backward_occurrences.get(event.seq_idx)
                if occurrence is None:
                    continue
                scope, anchor_seq = occurrence
                targets.setdefault(
                    (scope, anchor_seq, event.pp_stage, event.pp_mb_idx),
                    event,
                )

            if not targets:
                occurrence = forward_occurrences.get(producer_event.seq_idx)
                if occurrence is not None:
                    scope, anchor_seq = occurrence
                    targets[
                        (
                            scope,
                            anchor_seq,
                            producer_event.pp_stage,
                            producer_event.pp_mb_idx,
                        )
                    ] = producer_event
                else:
                    targets[
                        (
                            "<unattributed>",
                            producer_event.seq_idx,
                            producer_event.pp_stage,
                            producer_event.pp_mb_idx,
                        )
                    ] = producer_event
                    unattributed_count += 1
                    unattributed_bytes += lifetime.num_bytes

            for (scope, anchor_seq, pp_stage, pp_mb_idx), target_event in targets.items():
                context.activation_offload_tensors.append(
                    CheckpointTensorRecord(
                        checkpoint_id=f"part0:{scope}",
                        marker_kind=(
                            "layer" if scope != "<unattributed>" else "unattributed"
                        ),
                        seq_idx=anchor_seq,
                        tensor_id=lifetime.tensor_id,
                        role="activation_saved",
                        shape=ref.shape if ref is not None else lifetime.shape,
                        dtype=ref.dtype if ref is not None else lifetime.dtype,
                        num_bytes=lifetime.num_bytes,
                        modeled_num_bytes=0,
                        requires_grad=(
                            ref.requires_grad if ref is not None else False
                        ),
                        is_saved_activation=True,
                        residency_policy="offloaded",
                        pp_stage=pp_stage,
                        pp_mb_idx=pp_mb_idx,
                        comp_type=target_event.comp_type,
                    )
                )

        if offloaded_count:
            context.notes.append(
                "Activation-offload plugin removed "
                f"{offloaded_count} forward-to-backward tensors "
                f"({offloaded_bytes} logical bytes) from modeled device memory."
            )
        if unattributed_count:
            context.notes.append(
                "Activation-offload plugin could not map "
                f"{unattributed_count} tensors ({unattributed_bytes} logical bytes) "
                "to a transformer layer; they are reported under "
                "part0:<unattributed>."
            )
        return []
