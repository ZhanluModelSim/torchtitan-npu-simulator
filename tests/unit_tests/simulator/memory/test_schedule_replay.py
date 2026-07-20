# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from torchtitan_npu.simulator.ir.schedule_plan import ScheduleAction, SchedulePlan
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.export import export_memory_plan, memory_plan_to_chrome_trace
from torchtitan_npu.simulator.memory.schedule_replay import estimate_schedule_memory
from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent, RawMemoryEvent, TensorRef


def _ref(tensor_id: int, num_bytes: int = 100) -> TensorRef:
    return TensorRef(
        tensor_id=tensor_id,
        name=f"t{tensor_id}",
        shape=(num_bytes,),
        dtype="uint8",
        device="meta",
        num_bytes=num_bytes,
    )


def _event(
    seq_idx: int,
    op_id: int,
    *,
    comp_type: str,
    phase: str,
    inputs: tuple[TensorRef, ...] = (),
    outputs: tuple[TensorRef, ...] = (),
    module_path: str = "",
) -> RawMemoryEvent:
    return RawMemoryEvent(
        event_id=seq_idx,
        op_id=op_id,
        seq_idx=seq_idx,
        raw_op_type="aten.test.default",
        op_type="elementwise",
        phase=phase,
        module_path=module_path,
        inputs=inputs,
        outputs=outputs,
        execution_kind="original_forward" if phase == "forward" else "backward",
        pp_stage=0,
        pp_mb_idx=0,
        comp_type=comp_type,
    )


def _action(index: int, comp_type: str, microbatch: int) -> ScheduleAction:
    return ScheduleAction(
        id=index,
        action_id=f"a{index}",
        rank=0,
        stage=0,
        mb_idx=microbatch,
        action_type="COMPUTE",
        comp_type=comp_type,
        template_ref=f"s0_{comp_type}",
        seq_idx=(index + 1) * 10,
    )


def _plan(actions: list[ScheduleAction], *, pp_degree: int = 4, microbatches: int = 8) -> SchedulePlan:
    return SchedulePlan(
        plan_id="test",
        workload_type="train",
        step_templates={},
        actions=actions,
        data_slots={},
        pp_degree=pp_degree,
        num_micro_batches=microbatches,
        pipeline_schedule="test-schedule",
    )


def test_non_pp_keeps_original_estimator_path() -> None:
    output = _ref(10)
    events = [_event(3, 7, comp_type="F", phase="forward", outputs=(output,))]

    plan = estimate_schedule_memory(events, schedule_plan=_plan([], pp_degree=1, microbatches=1))
    baseline = estimate_static_memory(events)

    assert plan.to_dict() == baseline.to_dict()
    assert plan.action_spans == []
    assert not any("PP memory replay" in note for note in plan.notes)


def test_pp_replay_follows_stage0_warmup_steady_and_cooldown_order() -> None:
    activation = _ref(10)
    grad = _ref(20, 4)
    events = [
        _event(1, 101, comp_type="F", phase="forward", outputs=(activation,)),
        _event(20, 201, comp_type="B", phase="backward", inputs=(activation,), outputs=(grad,)),
    ]
    order = [
        ("F", 0), ("F", 1), ("F", 2), ("F", 3),
        ("B", 0), ("F", 4), ("B", 1), ("F", 5),
        ("B", 2), ("F", 6), ("B", 3), ("F", 7),
        ("B", 4), ("B", 5), ("B", 6), ("B", 7),
    ]
    actions = [_action(index, comp_type, mb) for index, (comp_type, mb) in enumerate(order)]

    plan = estimate_schedule_memory(events, schedule_plan=_plan(actions))

    assert [(span.comp_type, span.microbatch) for span in plan.action_spans] == order
    assert [(event.comp_type, event.pp_mb_idx) for event in plan.raw_events] == order
    activations = [lifetime for lifetime in plan.tensor_lifetimes if lifetime.kind == "activation"]
    assert len(activations) == 8
    assert len({lifetime.tensor_id for lifetime in activations}) == 8
    assert plan.peak_active_bytes >= 4 * activation.num_bytes


def test_pp_replay_uses_arbitrary_comp_types_and_flattens_overlap() -> None:
    forward = _event(1, 101, comp_type="F", phase="forward", outputs=(_ref(10),))
    backward_input = _event(2, 201, comp_type="I", phase="backward", outputs=(_ref(20),))
    backward_weight = _event(3, 301, comp_type="W", phase="backward", outputs=(_ref(30),))
    sub_f = _action(1, "F", 1)
    sub_i = _action(2, "I", 0)
    overlap = ScheduleAction(
        id=0,
        action_id="overlap",
        rank=0,
        stage=-1,
        mb_idx=-1,
        action_type="OVERLAP_F_B",
        sub_actions=[sub_f, sub_i],
    )
    actions = [_action(0, "F", 0), overlap, _action(3, "W", 0), _action(4, "W", 1)]

    plan = estimate_schedule_memory(
        [forward, backward_input, backward_weight],
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
    )

    assert [(span.comp_type, span.microbatch) for span in plan.action_spans] == [
        ("F", 0), ("F", 1), ("I", 0), ("W", 0), ("W", 1),
    ]


def test_pp_replay_gives_each_microbatch_distinct_external_stage_input() -> None:
    stage_input = _ref(1)
    output = _ref(2)
    grad = _ref(3)
    events = [
        _event(1, 101, comp_type="F", phase="forward", inputs=(stage_input,), outputs=(output,)),
        _event(2, 201, comp_type="B", phase="backward", inputs=(stage_input, output), outputs=(grad,)),
    ]
    actions = [
        _action(0, "F", 0), _action(1, "F", 1),
        _action(2, "B", 0), _action(3, "B", 1),
    ]

    plan = estimate_schedule_memory(
        events,
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
    )

    stage_inputs = [
        lifetime for lifetime in plan.tensor_lifetimes if lifetime.kind == "external_input"
    ]
    assert len(stage_inputs) == 2
    assert len({lifetime.tensor_id for lifetime in stage_inputs}) == 2
    assert all("backward" in lifetime.consumer_phases for lifetime in stage_inputs)


def test_pp_p2p_recv_is_represented_by_replayed_stage_input_not_double_counted() -> None:
    stage_input = _ref(1)
    output = _ref(2)
    recv = replace(
        _event(0, 50, comp_type="F", phase="forward", outputs=(stage_input,)),
        raw_op_type="comm.p2p_recv",
        op_type="p2p_recv",
    )
    forward = _event(
        1,
        101,
        comp_type="F",
        phase="forward",
        inputs=(stage_input,),
        outputs=(output,),
    )
    recv_action = ScheduleAction(
        id=0,
        action_id="recv",
        rank=0,
        stage=0,
        mb_idx=0,
        action_type="RECV_F",
        seq_idx=0,
        comm_op_id=50,
    )
    actions = [recv_action, _action(1, "F", 0), _action(2, "F", 1)]
    comm = SimpleNamespace(op_id=50, comm_layer="L2", p2p_direction="forward_recv")

    plan = estimate_schedule_memory(
        [recv, forward],
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
        comm_events=[comm],
    )

    assert sum(lifetime.kind == "external_input" for lifetime in plan.tensor_lifetimes) == 2
    assert not any(lifetime.kind == "comm_buffer" for lifetime in plan.tensor_lifetimes)


def test_stale_l2_comm_op_id_does_not_remove_compute_template() -> None:
    forward = _event(1, 101, comp_type="F", phase="forward", outputs=(_ref(10),))
    actions = [_action(0, "F", 0), _action(1, "F", 1)]
    stale_comm = SimpleNamespace(op_id=101, comm_layer="L2", p2p_direction="")

    plan = estimate_schedule_memory(
        [forward],
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
        comm_events=[stale_comm],
    )

    assert [(event.comp_type, event.pp_mb_idx) for event in plan.raw_events] == [
        ("F", 0),
        ("F", 1),
    ]


def test_pp_action_spans_are_exported_to_trace_and_csv(tmp_path) -> None:
    events = [_event(1, 101, comp_type="F", phase="forward", outputs=(_ref(10),))]
    actions = [_action(0, "F", 0), _action(1, "F", 1)]
    plan = estimate_schedule_memory(
        events,
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
    )

    trace = memory_plan_to_chrome_trace(plan)
    action_events = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("tid") == 100
    ]
    assert [event["name"] for event in action_events] == ["F mb0", "F mb1"]

    export_memory_plan(plan, str(tmp_path))
    actions_csv = tmp_path / "memory" / "memory_actions.csv"
    assert actions_csv.is_file()
    assert actions_csv.read_text().splitlines()[0].startswith("action_id,action_type,stage,microbatch")


def test_pp_replay_rejects_missing_compute_template() -> None:
    actions = [_action(0, "F", 0), _action(1, "B", 0)]

    with pytest.raises(RuntimeError, match="stage=0/comp_type=B"):
        estimate_schedule_memory(
            [_event(1, 101, comp_type="F", phase="forward", outputs=(_ref(10),))],
            schedule_plan=_plan(actions, pp_degree=2, microbatches=1),
        )


def test_checkpoint_internal_tensor_is_released_in_each_forward_instance() -> None:
    internal = _ref(10)
    output = _ref(11)
    grad = _ref(12)
    events = [
        _event(
            1,
            101,
            comp_type="F",
            phase="forward",
            outputs=(internal,),
            module_path="layers.0._checkpoint_wrapped_module.norm",
        ),
        _event(
            2,
            102,
            comp_type="F",
            phase="forward",
            inputs=(internal,),
            outputs=(output,),
            module_path="layers.0._checkpoint_wrapped_module",
        ),
        _event(20, 201, comp_type="B", phase="backward", inputs=(internal,), outputs=(grad,)),
    ]
    actions = [
        _action(0, "F", 0),
        _action(1, "F", 1),
        _action(2, "B", 0),
        _action(3, "B", 1),
    ]

    plan = estimate_schedule_memory(
        events,
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
    )

    checkpoint_temps = [
        lifetime for lifetime in plan.tensor_lifetimes if lifetime.kind == "checkpoint_recompute_temp"
    ]
    assert len(checkpoint_temps) == 2
    forward_ends = {
        span.microbatch: span.end_seq for span in plan.action_spans if span.comp_type == "F"
    }
    for lifetime in checkpoint_temps:
        producer = next(event for event in plan.raw_events if event.seq_idx == lifetime.birth_seq)
        assert lifetime.death_seq <= forward_ends[producer.pp_mb_idx]


def test_explicit_fsdp_residency_is_not_multiplied_by_microbatch_replay() -> None:
    events = [
        _event(10, 101, comp_type="F", phase="forward", outputs=(_ref(10),)),
        _event(30, 201, comp_type="B", phase="backward", outputs=(_ref(20),)),
    ]
    actions = [
        _action(0, "F", 0), _action(1, "F", 1),
        _action(2, "B", 0), _action(3, "B", 1),
    ]
    markers = [
        FSDPResidencyEvent("layer0", "alloc", 5, "forward", 1024),
        FSDPResidencyEvent("layer0", "free", 25, "backward", 1024),
    ]

    plan = estimate_schedule_memory(
        events,
        schedule_plan=_plan(actions, pp_degree=2, microbatches=2),
        fsdp_residency_events=markers,
    )

    full_params = [lifetime for lifetime in plan.tensor_lifetimes if lifetime.kind == "fsdp_full_param"]
    assert len(full_params) == 1
    assert full_params[0].num_bytes == 1024
    forward_spans = [span for span in plan.action_spans if span.comp_type == "F"]
    assert full_params[0].birth_seq == forward_spans[0].start_seq
    assert full_params[0].death_seq == forward_spans[-1].end_seq


def test_fsdp_markers_use_raw_comm_positions_not_plan_indices() -> None:
    unshard = replace(
        _event(5, 50, comp_type="F", phase="forward"),
        raw_op_type="comm.allgather",
        op_type="allgather",
    )
    forward = _event(10, 101, comp_type="F", phase="forward", outputs=(_ref(10),))
    reshard = replace(
        _event(20, 60, comp_type="B", phase="backward"),
        raw_op_type="comm.reduce_scatter",
        op_type="reduce_scatter",
    )
    backward = _event(30, 201, comp_type="B", phase="backward", outputs=(_ref(20),))
    unshard_action = ScheduleAction(
        id=0,
        action_id="unshard",
        rank=0,
        stage=0,
        mb_idx=-1,
        action_type="UNSHARD",
        seq_idx=0,
        comm_op_id=50,
    )
    reshard_action = ScheduleAction(
        id=2,
        action_id="reshard",
        rank=0,
        stage=0,
        mb_idx=-1,
        action_type="RESHARD",
        seq_idx=2,
        comm_op_id=60,
    )
    actions = [
        unshard_action,
        replace(_action(1, "F", 0), seq_idx=10),
        reshard_action,
        replace(_action(3, "B", 0), seq_idx=30),
    ]
    markers = [
        FSDPResidencyEvent("layer0", "alloc", 4, "forward", 1024),
        FSDPResidencyEvent("layer0", "free", 15, "forward", 1024),
    ]
    comm_events = [
        SimpleNamespace(op_id=50, comm_layer="L2", p2p_direction=""),
        SimpleNamespace(op_id=60, comm_layer="L2", p2p_direction=""),
    ]

    plan = estimate_schedule_memory(
        [unshard, forward, reshard, backward],
        schedule_plan=_plan(actions, pp_degree=2, microbatches=1),
        comm_events=comm_events,
        fsdp_residency_events=markers,
    )

    full_param = next(
        lifetime for lifetime in plan.tensor_lifetimes if lifetime.kind == "fsdp_full_param"
    )
    spans = {span.action_id: span for span in plan.action_spans}
    assert spans["unshard"].source_seq_idx == 5
    assert spans["reshard"].source_seq_idx == 20
    assert full_param.birth_seq == spans["unshard"].start_seq
    assert full_param.death_seq == spans["a1"].end_seq
