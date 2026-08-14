# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified saved-activation offload modeling."""

from __future__ import annotations

import re

from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_npu.simulator.memory.records import (
    AutogradSavedTensorEvent,
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
    """Model activations retained for normal backward.

    When capture observed autograd saved-tensor slots, they are the source of
    truth. The former use-def rule remains a compatibility fallback only.
    """

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        if (
            context.autograd_saved_tensors
            and not context.checkpoint_boundary_events
        ):
            return self._apply_exact_autograd_slots(context)
        if not context.offload_ac_saved_tensors:
            return []
        return self._apply_liveness_fallback(context)

    def _apply_exact_autograd_slots(
        self,
        context: MemoryModelContext,
    ) -> list[TensorLifetime]:
        """Use real autograd pack/unpack slots instead of use-def inference."""
        event_by_seq = {event.seq_idx: event for event in context.events}
        forward_occurrences = _layer_occurrences(context.events, phase="forward")
        backward_occurrences = _layer_occurrences(context.events, phase="backward")

        excluded_kinds = {
            "external_input",
            "fsdp_full_param",
            "comm_buffer",
            "parameter_shard",
            "checkpoint_recompute_temp",
        }
        chosen_by_storage: dict[str, tuple[AutogradSavedTensorEvent, TensorLifetime]] = {}
        for saved in context.autograd_saved_tensors:
            if (
                saved.phase != "forward"
                or saved.execution_kind != "original_forward"
                or saved.unpack_seq < 0
            ):
                continue
            tensor_id = _resolve_alias(saved.tensor_id, context.alias_base_by_tensor_id)
            lifetime = context.lifetimes_by_tensor_id.get(tensor_id)
            if lifetime is None or lifetime.kind in excluded_kinds:
                continue
            existing = chosen_by_storage.get(saved.storage_key)
            if existing is None or saved.unpack_seq > existing[0].unpack_seq:
                chosen_by_storage[saved.storage_key] = (saved, lifetime)

        selected_tensor_ids = {
            _resolve_alias(saved.tensor_id, context.alias_base_by_tensor_id)
            for saved, _lifetime in chosen_by_storage.values()
        }
        released_count = 0
        released_bytes = 0
        for tensor_id, lifetime in context.lifetimes_by_tensor_id.items():
            if (
                lifetime.producer_phase != "forward"
                or "backward" not in lifetime.consumer_phases
                or lifetime.kind in excluded_kinds
                or tensor_id in selected_tensor_ids
            ):
                continue
            forward_seqs = [
                seq_idx
                for seq_idx, phase in zip(
                    lifetime.consumer_seqs,
                    lifetime.consumer_phases,
                    strict=True,
                )
                if phase == "forward"
            ]
            lifetime.death_seq = max([lifetime.birth_seq, *forward_seqs])
            lifetime.kind = "temporary"
            lifetime.reason = "not_autograd_saved"
            released_count += 1
            released_bytes += lifetime.num_bytes

        recorded: set[int] = set()
        for saved, lifetime in chosen_by_storage.values():
            lifetime.death_seq = max(lifetime.birth_seq, saved.unpack_seq)
            lifetime.kind = "activation"
            lifetime.reason = "autograd_saved_tensor"
            if context.offload_ac_saved_tensors:
                lifetime.modeled_num_bytes = 0
                lifetime.residency_policy = "offloaded"
                lifetime.kind = "offloaded_activation"

            if not context.offload_ac_saved_tensors or id(lifetime) in recorded:
                continue
            recorded.add(id(lifetime))
            producer_event = event_by_seq.get(lifetime.birth_seq)
            if producer_event is None:
                continue
            ref = _producer_ref(
                _resolve_alias(saved.tensor_id, context.alias_base_by_tensor_id),
                producer_event,
                context.alias_base_by_tensor_id,
            )
            consumer_event = event_by_seq.get(saved.unpack_seq)
            occurrence = (
                backward_occurrences.get(saved.unpack_seq)
                if consumer_event is not None
                else None
            )
            if occurrence is None:
                occurrence = forward_occurrences.get(producer_event.seq_idx)
            if occurrence is None:
                scope, anchor_seq = "<unattributed>", producer_event.seq_idx
            else:
                scope, anchor_seq = occurrence
            target_event = consumer_event or producer_event
            context.activation_offload_tensors.append(
                CheckpointTensorRecord(
                    checkpoint_id=f"part0:{scope}",
                    marker_kind="layer" if scope != "<unattributed>" else "unattributed",
                    seq_idx=anchor_seq,
                    tensor_id=lifetime.tensor_id,
                    role="activation_saved",
                    shape=ref.shape if ref is not None else lifetime.shape,
                    dtype=ref.dtype if ref is not None else lifetime.dtype,
                    num_bytes=lifetime.num_bytes,
                    modeled_num_bytes=0,
                    requires_grad=ref.requires_grad if ref is not None else False,
                    is_saved_activation=True,
                    residency_policy="offloaded",
                    pp_stage=target_event.pp_stage,
                    pp_mb_idx=target_event.pp_mb_idx,
                    comp_type=target_event.comp_type,
                )
            )

        selected_bytes = sum(
            lifetime.num_bytes for _saved, lifetime in chosen_by_storage.values()
        )
        context.notes.append(
            "Autograd saved-tensor capture selected "
            f"{len(chosen_by_storage)} unique saved storages "
            f"({selected_bytes} logical bytes) for normal backward."
        )
        if released_count:
            context.notes.append(
                "Autograd saved-tensor capture released "
                f"{released_count} use-def-only forward tensors "
                f"({released_bytes} bytes) at their final forward consumer."
            )
        return []

    def _apply_liveness_fallback(
        self,
        context: MemoryModelContext,
    ) -> list[TensorLifetime]:
        """Compatibility path for checkpointed or unreplayed PP captures."""

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
