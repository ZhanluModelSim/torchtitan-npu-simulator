# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation-checkpoint liveness refinement.

``CheckpointWrapper`` executes its wrapped module again during backward. A
raw use-def scan cannot distinguish that recomputed use from a saved forward
activation, so it would retain every internal tensor until backward. The only
forward tensors that cross the checkpoint boundary are the wrapper inputs and
outputs; the rest can be released after their final forward consumer.
"""

from __future__ import annotations

from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_npu.simulator.memory.records import CheckpointTensorRecord, TensorLifetime

_CHECKPOINT_WRAPPED_MODULE = "._checkpoint_wrapped_module"


def _checkpoint_scope(module_path: str) -> str:
    """Return the wrapper path for an op inside a CheckpointWrapper."""
    prefix, separator, _ = module_path.partition(_CHECKPOINT_WRAPPED_MODULE)
    return prefix if separator else ""


def _resolve_alias(tensor_id: int, alias_base_by_tensor_id: dict[int, int]) -> int:
    seen: set[int] = set()
    current = tensor_id
    while current in alias_base_by_tensor_id and current not in seen:
        seen.add(current)
        current = alias_base_by_tensor_id[current]
    return current


class ActivationCheckpointPlugin(MemoryModelPlugin):
    """Release checkpoint-internal forward tensors before recomputation."""

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        event_by_seq = {event.seq_idx: event for event in context.events}
        released_count = 0
        released_bytes = 0

        for lifetime in context.lifetimes_by_tensor_id.values():
            if lifetime.producer_phase != "forward":
                continue
            producer_event = event_by_seq.get(lifetime.birth_seq)
            if producer_event is None:
                continue
            scope = _checkpoint_scope(producer_event.module_path)
            if not scope or "backward" not in lifetime.consumer_phases:
                continue

            forward_consumers = [
                event_by_seq[seq_idx]
                for seq_idx, phase in zip(lifetime.consumer_seqs, lifetime.consumer_phases)
                if phase == "forward" and seq_idx in event_by_seq
            ]
            crosses_checkpoint_boundary = any(
                event.module_path and _checkpoint_scope(event.module_path) != scope
                for event in forward_consumers
            )
            if crosses_checkpoint_boundary:
                continue

            lifetime.kind = "checkpoint_recompute_temp"
            lifetime.reason = "checkpoint_internal_recompute"
            lifetime.death_seq = max(
                lifetime.birth_seq,
                *(event.seq_idx for event in forward_consumers),
            )
            released_count += 1
            released_bytes += lifetime.num_bytes

        if released_count:
            context.notes.append(
                "Activation-checkpoint plugin released "
                f"{released_count} checkpoint-internal forward lifetimes "
                f"({released_bytes} bytes) before backward recomputation."
            )

        saved_tensor_ids: set[int] = set()
        untracked_saved_count = 0
        for boundary in context.checkpoint_boundary_events:
            for role, refs in (("input", boundary.inputs), ("output", boundary.outputs)):
                seen_refs: set[int] = set()
                for ref in refs:
                    if ref.tensor_id in seen_refs:
                        continue
                    seen_refs.add(ref.tensor_id)
                    root_tensor_id = _resolve_alias(
                        ref.tensor_id,
                        context.alias_base_by_tensor_id,
                    )
                    lifetime = context.lifetimes_by_tensor_id.get(root_tensor_id)
                    is_saved_activation = role == "input" and ref.requires_grad
                    if is_saved_activation and lifetime is not None:
                        lifetime.kind = "checkpoint_saved_activation"
                        lifetime.reason = "checkpoint_boundary_input"
                        saved_tensor_ids.add(root_tensor_id)
                        if context.offload_ac_saved_tensors:
                            lifetime.modeled_num_bytes = 0
                            lifetime.residency_policy = "offloaded"
                    elif is_saved_activation:
                        untracked_saved_count += 1

                    if is_saved_activation and context.offload_ac_saved_tensors:
                        modeled_num_bytes = 0
                        residency_policy = "offloaded"
                    elif lifetime is not None:
                        modeled_num_bytes = lifetime.resident_num_bytes
                        residency_policy = lifetime.residency_policy
                    else:
                        modeled_num_bytes = 0
                        residency_policy = "not_tracked"
                    context.checkpoint_tensors.append(
                        CheckpointTensorRecord(
                            checkpoint_id=boundary.checkpoint_id,
                            marker_kind=boundary.marker_kind,
                            seq_idx=boundary.seq_idx,
                            tensor_id=(
                                lifetime.tensor_id
                                if lifetime is not None
                                else f"tensor:{root_tensor_id}"
                            ),
                            role=role,
                            shape=ref.shape,
                            dtype=ref.dtype,
                            num_bytes=ref.num_bytes,
                            modeled_num_bytes=modeled_num_bytes,
                            requires_grad=ref.requires_grad,
                            is_saved_activation=is_saved_activation,
                            residency_policy=residency_policy,
                            pp_stage=boundary.pp_stage,
                            pp_mb_idx=boundary.pp_mb_idx,
                            comp_type=boundary.comp_type,
                        )
                    )

        saved_logical_bytes = sum(
            lifetime.num_bytes
            for tensor_id, lifetime in context.lifetimes_by_tensor_id.items()
            if tensor_id in saved_tensor_ids
        )
        if saved_tensor_ids:
            policy = (
                "offloaded from modeled device memory"
                if context.offload_ac_saved_tensors
                else "kept resident"
            )
            context.notes.append(
                "Activation-checkpoint boundary capture identified "
                f"{len(saved_tensor_ids)} saved activation tensors "
                f"({saved_logical_bytes} logical bytes); {policy}."
            )
        if untracked_saved_count:
            context.notes.append(
                f"{untracked_saved_count} checkpoint saved-activation records "
                "had no use-def lifetime and were retained as metadata only."
            )
        return []
