# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fallback gradient residency for parameters skipped by meta autograd.

Some simulator shims stand in for custom kernels without an Autograd kernel.
Their parameter gradients never materialize as tensors, even though real
training accumulates them and supplies them to the optimizer. This plugin
adds only those missing local parameter gradients; gradients observed by
dispatch remain owned by the generic use-def model.
"""

from __future__ import annotations

from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_npu.simulator.memory.records import TensorLifetime


def _checkpoint_scope(parameter_name: str) -> str:
    _, separator, parameter_name = parameter_name.partition(":")
    if not separator:
        return ""
    prefix, separator, _ = parameter_name.partition("._checkpoint_wrapped_module")
    return prefix if separator else ""


class MissingParameterGradientPlugin(MemoryModelPlugin):
    """Keep unobserved trainable gradients resident through optimizer step."""

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        if not context.missing_parameter_gradients:
            return []

        backward_events = [event for event in context.events if event.phase == "backward"]
        optimizer_events = [event for event in context.events if event.phase == "optimizer"]
        if not backward_events or not optimizer_events:
            return []

        fallback_birth = max(event.seq_idx for event in backward_events)
        optimizer_event = max(optimizer_events, key=lambda event: event.seq_idx)
        backward_by_scope: dict[str, int] = {}
        for event in backward_events:
            if "._checkpoint_wrapped_module" not in event.module_path:
                continue
            scope = event.module_path.partition("._checkpoint_wrapped_module")[0]
            backward_by_scope[scope] = max(backward_by_scope.get(scope, event.seq_idx), event.seq_idx)

        lifetimes: list[TensorLifetime] = []
        for gradient in context.missing_parameter_gradients:
            scope = _checkpoint_scope(gradient.name)
            birth_seq = backward_by_scope.get(scope, fallback_birth)
            lifetime = TensorLifetime(
                tensor_id=f"synthetic_grad:{gradient.name}",
                kind="gradient_accumulator",
                num_bytes=gradient.num_bytes,
                birth_seq=birth_seq,
                death_seq=optimizer_event.seq_idx,
                producer_op=-1,
                producer_raw_op="missing_parameter_grad",
                producer_phase="backward",
                shape=gradient.shape,
                dtype=gradient.dtype,
                reason="missing_meta_autograd",
            )
            lifetime.mark_consumer(optimizer_event.op_id, optimizer_event.seq_idx, "optimizer")
            lifetimes.append(lifetime)

        total_bytes = sum(item.num_bytes for item in lifetimes)
        context.notes.append(
            "Gradient residency plugin synthesized "
            f"{len(lifetimes)} missing parameter gradients ({total_bytes} bytes)."
        )
        return lifetimes
