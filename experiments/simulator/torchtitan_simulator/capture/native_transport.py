# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Annotate native transport facts without replacing outputs or autograd."""

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan_simulator.capture.op_mapping import communication_primitive


@dataclass(frozen=True, slots=True)
class NativeTransport:
    inputs: tuple[torch.Tensor, ...]
    group: Any
    outputs: tuple[torch.Tensor, ...] | None = None

    def attributes(self):
        return {
            "communication_input_bytes": sum(t.numel() * t.element_size() for t in self.inputs),
            "communication_input_shapes": [tuple(t.shape) for t in self.inputs],
        }


def transport_operands(func, args):
    raw = str(func)
    if communication_primitive(raw) is None:
        return None
    if raw.startswith(("_c10d_functional.", "_dtensor.")):
        return NativeTransport((args[0],), args[-1])
    if raw in {"c10d.allreduce_.default", "c10d.broadcast_.default"}:
        return NativeTransport(tuple(args[0]), args[1], tuple(args[0]))
    if raw == "c10d.alltoall_.default":
        return NativeTransport(tuple(args[1]), args[2], tuple(args[0]))
    if raw in {"c10d._allgather_base_.default", "c10d._reduce_scatter_base_.default", "c10d.alltoall_base_.default"}:
        return NativeTransport((args[1],), args[2], (args[0],))
    raise RuntimeError(f"Native communication schema needs an operand adapter: {func._schema}")


def transport_dependencies(func, operands):
    if operands is None:
        return ()
    primitive = communication_primitive(str(func))
    from torchtitan_simulator.capture.comm_events import _collective_dependencies
    from torchtitan_simulator.capture.fsdp_residency import get_active_fsdp_reduction_dependencies

    return (
        *_collective_dependencies.get(),
        *(
            get_active_fsdp_reduction_dependencies(operands.group)
            if primitive in {"allreduce", "reduce_scatter"}
            else ()
        ),
    )


def record_native_transport(func, operands, result, *, captured_op_id, source_op_id):
    from torchtitan_simulator.capture.comm_events import _record_comm_with_l0, get_active_recorder

    recorder = get_active_recorder()
    if recorder is None or operands is None or captured_op_id is None:
        return
    primitive = communication_primitive(str(func))
    assert primitive is not None
    outputs = (result,) if operands.outputs is None else operands.outputs
    _record_comm_with_l0(
        recorder,
        primitive,
        operands.group,
        operands.inputs[0],
        outputs[0],
        captured_op_id=captured_op_id,
        source_op_id=source_op_id,
        native_inputs=operands.inputs,
        native_outputs=outputs,
    )
