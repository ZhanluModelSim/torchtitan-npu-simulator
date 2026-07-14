# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.export import export_memory_plan, memory_plan_to_chrome_trace
from torchtitan_npu.simulator.memory.records import RawMemoryEvent, TensorRef


def tref(tensor_id: int, num_bytes: int = 16) -> TensorRef:
    return TensorRef(
        tensor_id=tensor_id,
        name=f"t{tensor_id}",
        shape=(num_bytes // 4,),
        dtype="float32",
        device="meta",
        num_bytes=num_bytes,
    )


def event(
    seq_idx: int,
    op_id: int,
    raw_op_type: str,
    *,
    inputs: list[TensorRef] | None = None,
    outputs: list[TensorRef] | None = None,
    phase: str = "forward",
    op_type: str = "elementwise",
) -> RawMemoryEvent:
    return RawMemoryEvent(
        event_id=seq_idx,
        op_id=op_id,
        seq_idx=seq_idx,
        raw_op_type=raw_op_type,
        op_type=op_type,
        phase=phase,
        module_path="",
        inputs=tuple(inputs or []),
        outputs=tuple(outputs or []),
    )


def test_simple_chain_frees_output_after_last_consumer():
    a, b, c = tref(1), tref(2), tref(3)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", inputs=[a], outputs=[b]),
        event(1, 11, "aten.sum.default", inputs=[b], outputs=[c]),
    ])
    b_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert b_lifetime.birth_seq == 0
    assert b_lifetime.death_seq == 1
    assert b_lifetime.kind == "temporary"


def test_forward_tensor_consumed_in_backward_is_activation():
    a, b, grad = tref(1), tref(2), tref(3)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", inputs=[a], outputs=[b], phase="forward"),
        event(5, 20, "aten.relu_backward.default", inputs=[b], outputs=[grad], phase="backward"),
    ])
    b_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert b_lifetime.kind == "activation"
    assert b_lifetime.death_seq == 5


def test_checkpoint_like_recompute_does_not_extend_original_forward_temp():
    x, y, z, grad = tref(1), tref(2), tref(3), tref(4)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", inputs=[x], outputs=[y], phase="forward"),
        event(1, 11, "aten.sum.default", inputs=[y], outputs=[z], phase="forward"),
        event(5, 20, "aten.relu.default", inputs=[x], outputs=[tref(5)], phase="backward"),
        event(6, 21, "aten.relu_backward.default", inputs=[tref(5)], outputs=[grad], phase="backward"),
    ])
    y_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert y_lifetime.kind == "temporary"
    assert y_lifetime.death_seq == 1


def test_alias_output_has_zero_bytes():
    a, b = tref(1, 64), tref(2, 64)
    plan = estimate_static_memory([
        event(0, 10, "aten.view.default", inputs=[a], outputs=[b]),
    ])
    alias = next(item for item in plan.tensor_lifetimes if item.tensor_id == "alias:2")
    assert alias.kind == "alias"
    assert alias.num_bytes == 0


@dataclass
class FakeComm:
    op_id: int
    comm_primitive: str
    comm_dim: str


def test_fsdp_allgather_output_is_classified_as_full_param_buffer():
    shard, full = tref(1, 32), tref(2, 128)
    plan = estimate_static_memory(
        [event(0, 10, "comm.allgather", inputs=[shard], outputs=[full], op_type="allgather")],
        comm_events=[FakeComm(op_id=10, comm_primitive="allgather", comm_dim="fsdp")],
    )
    full_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert full_lifetime.kind == "fsdp_full_param"
    assert full_lifetime.num_bytes == 128


def test_parameter_bytes_are_persistent_and_counted():
    model = nn.Linear(4, 2, bias=False, device="meta")
    plan = estimate_static_memory([], model_parts=[model])
    assert plan.persistent_param_bytes == 4 * 2 * 4
    assert plan.peak_active_bytes == plan.persistent_param_bytes


def test_memory_plan_exports_compact_chrome_trace(tmp_path):
    a, b, grad = tref(1, 32), tref(2, 32), tref(3, 32)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", inputs=[a], outputs=[b], phase="forward"),
        event(5, 20, "aten.relu_backward.default", inputs=[b], outputs=[grad], phase="backward"),
    ])

    trace = memory_plan_to_chrome_trace(plan)
    events = trace["traceEvents"]
    assert trace["displayTimeUnit"] == "ms"
    assert any(item["ph"] == "C" and item["name"] == "active_bytes" for item in events)
    assert any(item["ph"] == "X" and item["name"] == "forward" for item in events)
    assert any(item["ph"] == "X" and item["name"] == "backward" for item in events)
    assert any(item["ph"] == "i" and item["name"] == "peak active bytes" for item in events)

    export_memory_plan(plan, str(tmp_path))
    assert (tmp_path / "memory_trace.json").is_file()
