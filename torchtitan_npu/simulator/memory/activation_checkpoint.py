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
from torchtitan_npu.simulator.memory.records import TensorLifetime

_CHECKPOINT_WRAPPED_MODULE = "._checkpoint_wrapped_module"


def _checkpoint_scope(module_path: str) -> str:
    """Return the wrapper path for an op inside a CheckpointWrapper."""
    prefix, separator, _ = module_path.partition(_CHECKPOINT_WRAPPED_MODULE)
    return prefix if separator else ""


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
        return []
