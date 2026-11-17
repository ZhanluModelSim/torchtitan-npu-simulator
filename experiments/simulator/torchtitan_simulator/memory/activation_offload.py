# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified saved-activation offload modeling."""

from __future__ import annotations

from dataclasses import replace

from torchtitan_simulator.memory.layer_scope import activation_layer_scope
from torchtitan_simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_simulator.memory.records import (
    AutogradSavedTensorEvent,
    CheckpointTensorRecord,
    RawMemoryEvent,
    TensorLifetime,
    TensorRef,
)


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
        scope = activation_layer_scope(event.module_path)
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
        if phase == "backward" and seq_idx in event_by_seq and event_by_seq[seq_idx].execution_kind == "backward"
    ]


def _same_forward_action(saved: AutogradSavedTensorEvent, event: RawMemoryEvent | None) -> bool:
    """Require a known, matching semantic F action for a storage use."""
    return (
        event is not None
        and saved.comp_type == event.comp_type == "F"
        and saved.pp_stage == event.pp_stage
        and saved.pp_mb_idx == event.pp_mb_idx
    )


class ActivationOffloadPlugin(MemoryModelPlugin):
    """Model activations retained for normal backward.

    When capture observed autograd saved-tensor slots, they are the source of
    truth. The former use-def rule remains a compatibility fallback only.
    """

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        if context.autograd_saved_tensors:
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
        consumed_in_forward: dict[str, bool] = {}
        capture_end = max((event.seq_idx for event in context.events), default=0) + 1

        def saved_last_use(saved: AutogradSavedTensorEvent) -> int:
            # An unconsumed save has no observed release, including truncated
            # captures and retained graphs. Keep it live through the capture.
            release = saved.release_seq if saved.release_tracking else saved.unpack_seq
            return release if release >= 0 else capture_end

        for saved in context.autograd_saved_tensors:
            if saved.phase != "forward" or saved.execution_kind != "original_forward":
                continue
            tensor_id = context.tensor_root(saved.tensor_id, saved.storage_key)
            lifetime = context.lifetimes_by_tensor_id.get(tensor_id)
            if lifetime is None or lifetime.kind in excluded_kinds:
                continue
            consumed_in_forward[saved.storage_key] = consumed_in_forward.get(saved.storage_key, True) and (
                (unpack := event_by_seq.get(saved.unpack_seq)) is not None
                and unpack.execution_kind == "backward"
                and _same_forward_action(saved, unpack)
                and all(_same_forward_action(saved, event_by_seq.get(seq)) for seq in lifetime.consumer_seqs)
            )
            existing = chosen_by_storage.get(saved.storage_key)
            if existing is None or saved_last_use(saved) > saved_last_use(existing[0]):
                chosen_by_storage[saved.storage_key] = (saved, lifetime)

        selected_tensor_ids = {
            context.tensor_root(saved.tensor_id, saved.storage_key) for saved, _lifetime in chosen_by_storage.values()
        }
        retained_count = 0
        retained_bytes = 0
        for tensor_id, lifetime in context.lifetimes_by_tensor_id.items():
            if (
                lifetime.producer_phase != "forward"
                or "backward" not in lifetime.consumer_phases
                or lifetime.kind in excluded_kinds
                or tensor_id in selected_tensor_ids
            ):
                continue
            # Slots observe save_for_backward, not ordinary Python references.
            # A recorded backward consumer is positive retention evidence.
            lifetime.reason = "backward_use_without_saved_slot"
            retained_count += 1
            retained_bytes += lifetime.num_bytes

        checkpoint_recorded_ids = {item.tensor_id for item in context.checkpoint_tensors if item.is_saved_activation}
        recorded: set[int] = set()
        resident_sizes: dict[str, int] = {}
        for saved, lifetime in chosen_by_storage.values():
            lifetime.death_seq = max(
                lifetime.death_seq,
                lifetime.birth_seq,
                saved_last_use(saved),
                *lifetime.consumer_seqs,
            )
            if lifetime.kind not in {"checkpoint_saved_activation", "checkpoint_saved_for_recompute"}:
                lifetime.kind = "activation"
                lifetime.reason = "autograd_saved_tensor"
            producer_event = event_by_seq.get(lifetime.birth_seq)
            consumer_event = event_by_seq.get(saved.unpack_seq)
            saved_scope = activation_layer_scope(saved.module_path)
            if saved_scope:
                occurrence = (saved_scope, saved.pack_seq)
            elif saved.module_path:
                # An explicit head/norm/root save is outside decoder layers;
                # a stale backward path must not turn it into a layer save.
                occurrence = None
            else:
                occurrence = backward_occurrences.get(saved.unpack_seq)
                if occurrence is None:
                    for seq in lifetime.consumer_seqs:
                        occurrence = backward_occurrences.get(seq)
                        if occurrence is not None:
                            consumer_event = event_by_seq[seq]
                            break
                if occurrence is None and producer_event is not None:
                    occurrence = forward_occurrences.get(producer_event.seq_idx)
            known_owner = occurrence is not None or lifetime.tensor_id in checkpoint_recorded_ids
            if consumed_in_forward[saved.storage_key] or not known_owner:
                # Chunked loss saves real tensors, but consumes them before
                # decoder backward. Keep their observed device lifetimes;
                # being saved does not make them offload candidates. Head,
                # norm and unowned saves also stay resident by policy.
                lifetime.modeled_num_bytes = None
                lifetime.residency_policy = "resident"
                resident_sizes[lifetime.tensor_id] = lifetime.num_bytes
                continue
            if context.offload_ac_saved_tensors:
                lifetime.modeled_num_bytes = 0
                lifetime.residency_policy = "offloaded"
                if lifetime.kind == "activation":
                    lifetime.kind = "offloaded_activation"

            if (
                not context.offload_ac_saved_tensors
                or id(lifetime) in recorded
                or lifetime.tensor_id in checkpoint_recorded_ids
            ):
                continue
            recorded.add(id(lifetime))
            if producer_event is None:
                continue
            ref = _producer_ref(
                context.tensor_root(saved.tensor_id, saved.storage_key),
                producer_event,
                context.alias_base_by_tensor_id,
            )
            assert occurrence is not None
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

        if resident_sizes:
            context.checkpoint_tensors[:] = [
                replace(record, residency_policy="resident", modeled_num_bytes=resident_sizes[record.tensor_id])
                if record.tensor_id in resident_sizes and record.residency_policy == "offloaded"
                else record
                for record in context.checkpoint_tensors
            ]
            context.notes.append(
                f"Kept {len(resident_sizes)} chunk-local or non-layer/unattributed saved storages "
                "resident; they are excluded from activation offload payloads."
            )

        selected_bytes = sum(lifetime.num_bytes for _saved, lifetime in chosen_by_storage.values())
        context.notes.append(
            "Autograd saved-tensor capture selected "
            f"{len(chosen_by_storage)} unique saved storages "
            f"({selected_bytes} logical bytes) observed for backward."
        )
        if retained_count:
            context.notes.append(
                "Autograd saved-tensor capture retained "
                f"{retained_count} forward tensors with observed backward use "
                f"({retained_bytes} bytes) outside saved-tensor slots."
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
        checkpoint_recorded_ids = {item.tensor_id for item in context.checkpoint_tensors if item.is_saved_activation}

        offloaded_count = 0
        offloaded_bytes = 0
        unattributed_count = 0
        unattributed_bytes = 0

        for tensor_id, lifetime in context.lifetimes_by_tensor_id.items():
            if (
                lifetime.producer_phase != "forward"
                or "backward" not in lifetime.consumer_phases
                or lifetime.kind
                in {
                    "external_input",
                    "fsdp_full_param",
                    "checkpoint_recompute_temp",
                }
            ):
                continue

            producer_event = event_by_seq.get(lifetime.birth_seq)
            if producer_event is None:
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
                elif lifetime.tensor_id not in checkpoint_recorded_ids:
                    unattributed_count += 1
                    unattributed_bytes += lifetime.num_bytes
                    continue

            lifetime.modeled_num_bytes = 0
            lifetime.residency_policy = "offloaded"
            if lifetime.kind not in {"checkpoint_saved_activation", "checkpoint_saved_for_recompute"}:
                lifetime.kind = "offloaded_activation"
                lifetime.reason = "forward_tensor_saved_for_backward_offloaded"
            offloaded_count += 1
            offloaded_bytes += lifetime.num_bytes
            if lifetime.tensor_id in checkpoint_recorded_ids:
                continue

            for (scope, anchor_seq, pp_stage, pp_mb_idx), target_event in targets.items():
                context.activation_offload_tensors.append(
                    CheckpointTensorRecord(
                        checkpoint_id=f"part0:{scope}",
                        marker_kind=("layer" if scope != "<unattributed>" else "unattributed"),
                        seq_idx=anchor_seq,
                        tensor_id=lifetime.tensor_id,
                        role="activation_saved",
                        shape=ref.shape if ref is not None else lifetime.shape,
                        dtype=ref.dtype if ref is not None else lifetime.dtype,
                        num_bytes=lifetime.num_bytes,
                        modeled_num_bytes=0,
                        requires_grad=(ref.requires_grad if ref is not None else False),
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
                "to a transformer layer; they remain resident and are not offloaded."
            )
        return []
