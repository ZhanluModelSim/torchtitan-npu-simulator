# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.memory import estimator
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.export import export_memory_plan, memory_plan_to_chrome_trace
from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent, RawMemoryEvent, TensorRef


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
    module_path: str = "",
) -> RawMemoryEvent:
    return RawMemoryEvent(
        event_id=seq_idx,
        op_id=op_id,
        seq_idx=seq_idx,
        raw_op_type=raw_op_type,
        op_type=op_type,
        phase=phase,
        module_path=module_path,
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


def test_checkpoint_plugin_releases_internal_forward_tensor_before_backward():
    x, internal, output, grad = tref(1), tref(2), tref(3), tref(4)
    plan = estimate_static_memory([
        event(
            0,
            10,
            "aten.relu.default",
            inputs=[x],
            outputs=[internal],
            module_path="layers.0._checkpoint_wrapped_module.norm",
        ),
        event(
            1,
            11,
            "aten.add.Tensor",
            inputs=[internal],
            outputs=[output],
            module_path="layers.0._checkpoint_wrapped_module",
        ),
        event(5, 20, "aten.relu_backward.default", inputs=[internal], outputs=[grad], phase="backward"),
    ])

    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert lifetime.kind == "checkpoint_recompute_temp"
    assert lifetime.death_seq == 1


def test_checkpoint_plugin_keeps_cross_scope_output_as_activation():
    x, output, out, grad = tref(1), tref(2), tref(3), tref(4)
    plan = estimate_static_memory([
        event(
            0,
            10,
            "aten.relu.default",
            inputs=[x],
            outputs=[output],
            module_path="layers.0._checkpoint_wrapped_module",
        ),
        event(
            1,
            11,
            "aten.add.Tensor",
            inputs=[output],
            outputs=[out],
            module_path="layers.1._checkpoint_wrapped_module",
        ),
        event(5, 20, "aten.relu_backward.default", inputs=[output], outputs=[grad], phase="backward"),
    ])

    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert lifetime.kind == "activation"
    assert lifetime.death_seq == 5


def test_checkpoint_plugin_treats_pathless_collective_as_internal_transport():
    x, internal, comm_out, grad = tref(1), tref(2), tref(3), tref(4)
    plan = estimate_static_memory([
        event(
            0,
            10,
            "aten.relu.default",
            inputs=[x],
            outputs=[internal],
            module_path="layers.0._checkpoint_wrapped_module.moe",
        ),
        event(1, 11, "comm.all_to_all", inputs=[internal], outputs=[comm_out]),
        event(5, 20, "aten.relu_backward.default", inputs=[internal], outputs=[grad], phase="backward"),
    ])

    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert lifetime.kind == "checkpoint_recompute_temp"
    assert lifetime.death_seq == 1


def test_backward_output_consumed_by_optimizer_is_gradient_accumulator():
    grad = tref(2, 64)
    plan = estimate_static_memory([
        event(0, 10, "aten.mm.default", outputs=[grad], phase="backward"),
        event(5, 20, "optimizer.step", inputs=[grad], phase="optimizer"),
    ])

    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert lifetime.kind == "gradient_accumulator"
    assert lifetime.death_seq == 5


def test_alias_output_has_zero_bytes():
    a, b = tref(1, 64), tref(2, 64)
    plan = estimate_static_memory([
        event(0, 10, "aten.view.default", inputs=[a], outputs=[b]),
    ])
    alias = next(item for item in plan.tensor_lifetimes if item.tensor_id == "alias:2")
    assert alias.kind == "alias"
    assert alias.num_bytes == 0


def test_alias_consumer_extends_base_lifetime():
    x, base, view, out = tref(1), tref(2, 64), tref(3, 64), tref(4)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", inputs=[x], outputs=[base]),
        event(1, 11, "aten.view.default", inputs=[base], outputs=[view]),
        event(5, 12, "aten.sum.default", inputs=[view], outputs=[out]),
    ])
    base_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert base_lifetime.death_seq == 5
    assert base_lifetime.consumer_ops[-1] == 12


def test_parameter_alias_is_not_counted_as_external_input():
    model = nn.Linear(4, 8, device="meta")
    capture = OpDispatchCapture()
    x = torch.randn(2, 4, device="meta")
    with capture:
        model(x)

    plan = estimate_static_memory(capture.memory_events(), model_parts=[model])
    external_bytes = sum(item.num_bytes for item in plan.tensor_lifetimes if item.kind == "external_input")
    # Only the model input should be external. The transposed weight alias
    # consumed by addmm must resolve back to parameter_shard and remain zero.
    assert external_bytes == 2 * 4 * 4


@dataclass
class FakeComm:
    op_id: int
    comm_primitive: str
    comm_dim: str
    volume_bytes: int = 0
    world_size: int = 1
    dst_entry_op: int = 0
    comm_layer: str = ""


def test_fsdp_allgather_output_is_classified_as_full_param_buffer():
    shard, full = tref(1, 32), tref(2, 128)
    plan = estimate_static_memory(
        [event(0, 10, "comm.allgather", inputs=[shard], outputs=[full], op_type="allgather")],
        comm_events=[FakeComm(op_id=10, comm_primitive="allgather", comm_dim="fsdp")],
    )
    full_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert full_lifetime.kind == "fsdp_full_param"
    assert full_lifetime.num_bytes == 128
    assert not any(item.tensor_id.startswith("fsdp_full_param:") for item in plan.tensor_lifetimes)


def test_fsdp_residency_plugin_synthesizes_missing_full_param_lifetime():
    shard = tref(1, 32)
    plan = estimate_static_memory(
        [event(0, 10, "comm.allgather", inputs=[shard], outputs=[], op_type="allgather")],
        comm_events=[FakeComm(op_id=10, comm_primitive="allgather", comm_dim="fsdp", volume_bytes=32, world_size=4)],
    )

    full_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "fsdp_full_param:10")
    assert full_lifetime.kind == "fsdp_full_param"
    assert full_lifetime.num_bytes == 128
    assert full_lifetime.birth_seq == 0
    assert full_lifetime.death_seq == 0
    assert "FSDP residency plugin synthesized 1 full-param lifetimes" in " ".join(plan.notes)


def test_fsdp_residency_plugin_uses_comm_dst_entry_op_as_consumer():
    shard = tref(1, 32)
    x, y = tref(2, 16), tref(3, 16)
    plan = estimate_static_memory(
        [
            event(0, 10, "comm.allgather", inputs=[shard], outputs=[], op_type="allgather"),
            event(4, 20, "aten.mm.default", inputs=[x], outputs=[y]),
        ],
        comm_events=[
            FakeComm(
                op_id=10,
                comm_primitive="allgather",
                comm_dim="fsdp",
                volume_bytes=32,
                world_size=4,
                dst_entry_op=20,
            )
        ],
    )

    full_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "fsdp_full_param:10")
    assert full_lifetime.death_seq == 4
    assert full_lifetime.consumer_ops == [20]


def test_fsdp_explicit_markers_replace_full_param_and_bound_staging_buffer():
    shard, staging, full, out = tref(1, 32), tref(2, 128), tref(3, 128), tref(4, 16)
    plan = estimate_static_memory(
        [
            event(0, 10, "comm.allgather", inputs=[shard], outputs=[staging], op_type="allgather"),
            event(3, 20, "aten.mm.default", inputs=[full], outputs=[out]),
            event(50, 30, "aten.sum.default", inputs=[staging], outputs=[tref(5)]),
        ],
        comm_events=[
            FakeComm(op_id=10, comm_primitive="allgather", comm_dim="0", comm_layer="L2")
        ],
        fsdp_residency_events=[
            FSDPResidencyEvent("layer0", "alloc", 2, "forward", 128, (full.tensor_id,)),
            FSDPResidencyEvent("layer0", "free", 5, "forward", 128, (full.tensor_id,)),
        ],
    )

    residency = next(item for item in plan.tensor_lifetimes if item.kind == "fsdp_full_param")
    staging_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert (residency.birth_seq, residency.death_seq) == (2, 5)
    assert staging_lifetime.death_seq == 2
    assert staging_lifetime.reason == "fsdp_allgather_staging"
    assert not any(item.tensor_id == "external:3" for item in plan.tensor_lifetimes)


def test_parameter_bytes_are_persistent_and_counted():
    model = nn.Linear(4, 2, bias=False, device="meta")
    plan = estimate_static_memory([], model_parts=[model])
    assert plan.persistent_param_bytes == 4 * 2 * 4
    assert plan.peak_active_bytes == plan.persistent_param_bytes


def test_parameter_snapshot_deduplicates_by_parameter_identity(monkeypatch):
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False, device="meta"),
        nn.Linear(4, 4, bias=False, device="meta"),
    )
    shared_local_tensor = torch.empty(4, 4, device="meta")
    monkeypatch.setattr(estimator, "_to_local_tensor", lambda _param: shared_local_tensor)

    plan = estimate_static_memory([], model_parts=[model])

    assert plan.persistent_param_bytes == 2 * 4 * 4 * 4


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
    memory_dir = tmp_path / "memory"
    assert (memory_dir / "memory_trace.json").is_file()
    assert not (tmp_path / "memory_trace.json").exists()
    memory_events_header = (memory_dir / "memory_events.csv").read_text().splitlines()[0]
    memory_timeline_header = (memory_dir / "memory_timeline.csv").read_text().splitlines()[0]
    assert memory_events_header.startswith("event_id,seq_idx,phase,op_id")
    assert memory_timeline_header.startswith("seq_idx,phase,op_id,action")


def test_chrome_trace_includes_fsdp_full_param_counter():
    shard = tref(1, 32)
    plan = estimate_static_memory(
        [event(0, 10, "comm.allgather", inputs=[shard], outputs=[], op_type="allgather")],
        comm_events=[FakeComm(op_id=10, comm_primitive="allgather", comm_dim="fsdp", volume_bytes=32, world_size=4)],
    )

    trace = memory_plan_to_chrome_trace(plan)
    events = trace["traceEvents"]
    assert any(item["ph"] == "M" and item["args"].get("name") == "fsdp full-param bytes" for item in events)
    assert any(item["ph"] == "C" and item["name"] == "active_fsdp_full_param_bytes" for item in events)


def test_chrome_trace_includes_gradient_accumulator_counter():
    grad = tref(2, 64)
    plan = estimate_static_memory([
        event(0, 10, "aten.mm.default", outputs=[grad], phase="backward"),
        event(5, 20, "optimizer.step", inputs=[grad], phase="optimizer"),
    ])

    events = memory_plan_to_chrome_trace(plan)["traceEvents"]
    assert any(item["ph"] == "M" and item["args"].get("name") == "gradient accumulator bytes" for item in events)
    assert any(item["ph"] == "C" and item["name"] == "active_gradient_accumulator_bytes" for item in events)


def test_chrome_trace_starts_with_persistent_parameter_bytes():
    model = nn.Linear(4, 2, device="meta")
    plan = estimate_static_memory([], model_parts=[model])

    counters = [
        item for item in memory_plan_to_chrome_trace(plan)["traceEvents"]
        if item["ph"] == "C" and item["name"] == "active_bytes" and item["ts"] == 0
    ]

    assert counters[-1]["args"]["active_bytes"] == plan.persistent_param_bytes
