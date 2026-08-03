# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.memory import estimator
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.export import (
    export_memory_plan,
    export_memory_summary,
    memory_plan_to_chrome_trace,
)
from torchtitan_npu.simulator.memory.records import (
    CheckpointBoundaryEvent,
    FSDPResidencyEvent,
    RawMemoryEvent,
    TensorRef,
)


def tref(tensor_id: int, num_bytes: int = 16) -> TensorRef:
    return TensorRef(
        tensor_id=tensor_id,
        name=f"t{tensor_id}",
        shape=(num_bytes // 4,),
        dtype="float32",
        device="meta",
        num_bytes=num_bytes,
        requires_grad=True,
    )


def event(
    seq_idx: int,
    op_id: int,
    raw_op_type: str,
    *,
    inputs: list[TensorRef] | None = None,
    outputs: list[TensorRef] | None = None,
    phase: str = "forward",
    execution_kind: str | None = None,
    op_type: str = "elementwise",
    module_path: str = "",
    pp_stage: int = -1,
    pp_mb_idx: int = -1,
    comp_type: str = "",
) -> RawMemoryEvent:
    return RawMemoryEvent(
        event_id=seq_idx,
        op_id=op_id,
        seq_idx=seq_idx,
        raw_op_type=raw_op_type,
        op_type=op_type,
        phase=phase,
        execution_kind=execution_kind or {
            "forward": "original_forward",
            "backward": "backward",
            "optimizer": "optimizer",
        }[phase],
        module_path=module_path,
        inputs=tuple(inputs or []),
        outputs=tuple(outputs or []),
        pp_stage=pp_stage,
        pp_mb_idx=pp_mb_idx,
        comp_type=comp_type,
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
    release = next(item for item in plan.timeline_events if item.tensor_id == "tensor:2" and item.action == "free")
    assert release.phase == "forward"


def test_activation_offload_covers_none_mode_and_reports_each_layer():
    x = tref(1)
    layer0_saved = tref(2, 64)
    layer1_saved = tref(3, 128)
    grad1 = tref(4, 128)
    grad0 = tref(5, 64)
    events = [
        event(
            0,
            10,
            "aten.mm.default",
            inputs=[x],
            outputs=[layer0_saved],
            module_path="layers.0.attention",
        ),
        event(
            1,
            11,
            "aten.mm.default",
            inputs=[layer0_saved],
            outputs=[layer1_saved],
            module_path="layers.1.attention",
        ),
        event(
            5,
            20,
            "aten.mm.default",
            inputs=[layer1_saved],
            outputs=[grad1],
            phase="backward",
            execution_kind="backward",
            module_path="layers.1.attention",
        ),
        event(
            6,
            21,
            "aten.mm.default",
            inputs=[layer0_saved],
            outputs=[grad0],
            phase="backward",
            execution_kind="backward",
            module_path="layers.0.attention",
        ),
    ]

    resident_plan = estimate_static_memory(events)
    offloaded_plan = estimate_static_memory(
        events,
        offload_ac_saved_tensors=True,
    )

    offloaded = {
        item.tensor_id: item
        for item in offloaded_plan.tensor_lifetimes
        if item.kind == "offloaded_activation"
    }
    assert set(offloaded) == {"tensor:2", "tensor:3"}
    assert all(item.resident_num_bytes == 0 for item in offloaded.values())
    assert (
        offloaded_plan.model_active_bytes_peak
        < resident_plan.model_active_bytes_peak
    )

    summary = offloaded_plan.to_summary_dict()
    assert summary["activation_offload_tensor_count"] == 2
    assert summary["activation_offload_logical_bytes"] == 192
    assert summary["activation_offload_modeled_bytes"] == 0
    assert summary["activation_prefetch_tensor_count"] == 2
    assert summary["activation_prefetch_logical_bytes"] == 192
    assert summary["activation_prefetch_unattributed_tensor_count"] == 0
    assert summary["activation_prefetch"]["part0:layers.0"][
        "logical_bytes_per_instance"
    ] == 64
    assert summary["activation_prefetch"]["part0:layers.1"][
        "logical_bytes_per_instance"
    ] == 128

    records = {
        item.tensor_id: item
        for item in offloaded_plan.activation_offload_tensors
    }
    assert records["tensor:2"].checkpoint_id == "part0:layers.0"
    assert records["tensor:3"].checkpoint_id == "part0:layers.1"
    assert all(item.role == "activation_saved" for item in records.values())


def test_activation_offload_preserves_pipeline_microbatch_instances():
    saved0 = tref(2, 64)
    saved1 = tref(3, 64)
    plan = estimate_static_memory(
        [
            event(
                0,
                10,
                "aten.mm.default",
                inputs=[tref(1)],
                outputs=[saved0],
                module_path="layers.0.attention",
                pp_stage=1,
                pp_mb_idx=0,
                comp_type="F",
            ),
            event(
                1,
                11,
                "aten.mm.default",
                inputs=[tref(4)],
                outputs=[saved1],
                module_path="layers.0.attention",
                pp_stage=1,
                pp_mb_idx=1,
                comp_type="F",
            ),
            event(
                5,
                20,
                "aten.mm.default",
                inputs=[saved1],
                phase="backward",
                execution_kind="backward",
                module_path="layers.0.attention",
                pp_stage=1,
                pp_mb_idx=1,
                comp_type="B",
            ),
            event(
                6,
                21,
                "aten.mm.default",
                inputs=[saved0],
                phase="backward",
                execution_kind="backward",
                module_path="layers.0.attention",
                pp_stage=1,
                pp_mb_idx=0,
                comp_type="B",
            ),
        ],
        offload_ac_saved_tensors=True,
    )

    marker = plan.to_summary_dict()["activation_prefetch"][
        "part0:layers.0"
    ]
    assert marker["logical_bytes_per_instance"] == 64
    assert marker["logical_bytes_total"] == 128
    assert marker["instance_count"] == 2
    assert marker["pp_stages"] == [1]
    assert marker["microbatches"] == [0, 1]


@pytest.mark.parametrize(
    ("offload", "modeled_bytes", "has_timeline"),
    [(False, 64, True), (True, 0, False)],
)
def test_checkpoint_plugin_keeps_tensor_reused_during_recompute(
    offload,
    modeled_bytes,
    has_timeline,
):
    x, saved, output, recomputed = (
        tref(1),
        tref(2, 64),
        tref(3),
        tref(4),
    )
    plan = estimate_static_memory(
        [
            event(
                0,
                10,
                "aten.mm.default",
                inputs=[x],
                outputs=[saved],
                module_path="layers.0._checkpoint_wrapped_module.attention",
            ),
            event(
                1,
                11,
                "aten.add.Tensor",
                inputs=[saved],
                outputs=[output],
                module_path="layers.0._checkpoint_wrapped_module",
            ),
            event(
                5,
                20,
                "aten.bmm.default",
                inputs=[saved],
                outputs=[recomputed],
                phase="backward",
                execution_kind="recompute",
                module_path="layers.0._checkpoint_wrapped_module.attention",
            ),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=1,
                inputs=(x,),
                outputs=(output,),
                pp_stage=0,
                pp_mb_idx=0,
            )
        ],
        offload_ac_saved_tensors=offload,
    )

    lifetime = next(
        item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2"
    )
    assert lifetime.kind == "checkpoint_saved_for_recompute"
    assert lifetime.death_seq == 5
    assert lifetime.resident_num_bytes == modeled_bytes
    assert (
        any(item.tensor_id == "tensor:2" for item in plan.timeline_events)
        is has_timeline
    )

    summary = plan.to_summary_dict()
    assert summary["checkpoint_recompute_saved_tensor_count"] == 1
    assert summary["checkpoint_recompute_saved_logical_bytes"] == 64
    assert summary["checkpoint_recompute_saved_modeled_bytes"] == modeled_bytes
    assert summary["checkpoint_prefetch_unattributed_tensor_count"] == 0
    assert summary["checkpoint_prefetch_unattributed_logical_bytes"] == 0
    assert summary["checkpoint_saved_activations"]["part0:layers.0"][
        "logical_bytes_per_instance"
    ] == 16
    prefetch = summary["checkpoint_prefetch"]["part0:layers.0"]
    assert prefetch["boundary_logical_bytes_per_instance"] == 16
    assert prefetch["recompute_saved_logical_bytes_per_instance"] == 64
    assert prefetch["prefetch_logical_bytes_per_instance"] == 80
    assert prefetch["boundary_modeled_bytes_per_instance"] == (
        0 if offload else 16
    )
    assert prefetch["recompute_saved_modeled_bytes_per_instance"] == modeled_bytes
    assert prefetch["modeled_bytes_per_instance"] == (
        0 if offload else 80
    )
    assert prefetch["boundary_tensor_count_per_instance"] == 1
    assert prefetch["recompute_saved_tensor_count_per_instance"] == 1
    assert prefetch["tensor_count_per_instance"] == 2
    if offload:
        activation_prefetch = summary["activation_prefetch"][
            "part0:layers.0"
        ]
        assert activation_prefetch["logical_bytes_per_instance"] == 80
        assert activation_prefetch["modeled_bytes_per_instance"] == 0
        assert activation_prefetch["tensor_count_per_instance"] == 2
    else:
        assert summary["activation_prefetch"] == {}

    recompute_record = next(
        item
        for item in plan.checkpoint_tensors
        if item.role == "recompute_saved"
    )
    assert recompute_record.checkpoint_id == "part0:layers.0"
    assert recompute_record.seq_idx == 1
    assert recompute_record.num_bytes == 64
    assert recompute_record.modeled_num_bytes == modeled_bytes


def test_checkpoint_prefetch_matches_recompute_saved_tensors_to_microbatches():
    x0, saved0, output0 = tref(1), tref(2, 64), tref(3)
    x1, saved1, output1 = tref(4), tref(5, 64), tref(6)
    plan = estimate_static_memory(
        [
            event(
                0,
                10,
                "aten.mm.default",
                inputs=[x0],
                outputs=[saved0],
                module_path="layers.0._checkpoint_wrapped_module.attention",
                pp_stage=1,
                pp_mb_idx=0,
                comp_type="F",
            ),
            event(
                1,
                11,
                "aten.add.Tensor",
                inputs=[saved0],
                outputs=[output0],
                module_path="layers.0._checkpoint_wrapped_module",
                pp_stage=1,
                pp_mb_idx=0,
                comp_type="F",
            ),
            event(
                2,
                12,
                "aten.mm.default",
                inputs=[x1],
                outputs=[saved1],
                module_path="layers.0._checkpoint_wrapped_module.attention",
                pp_stage=1,
                pp_mb_idx=1,
                comp_type="F",
            ),
            event(
                3,
                13,
                "aten.add.Tensor",
                inputs=[saved1],
                outputs=[output1],
                module_path="layers.0._checkpoint_wrapped_module",
                pp_stage=1,
                pp_mb_idx=1,
                comp_type="F",
            ),
            event(
                5,
                20,
                "aten.bmm.default",
                inputs=[saved1],
                phase="backward",
                execution_kind="recompute",
                module_path="layers.0._checkpoint_wrapped_module.attention",
                pp_stage=1,
                pp_mb_idx=1,
                comp_type="B",
            ),
            event(
                6,
                21,
                "aten.bmm.default",
                inputs=[saved0],
                phase="backward",
                execution_kind="recompute",
                module_path="layers.0._checkpoint_wrapped_module.attention",
                pp_stage=1,
                pp_mb_idx=0,
                comp_type="B",
            ),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=1,
                inputs=(x0,),
                outputs=(output0,),
                pp_stage=1,
                pp_mb_idx=0,
            ),
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=3,
                inputs=(x1,),
                outputs=(output1,),
                pp_stage=1,
                pp_mb_idx=1,
            ),
        ],
        offload_ac_saved_tensors=True,
    )

    records = {
        item.tensor_id: item
        for item in plan.checkpoint_tensors
        if item.role == "recompute_saved"
    }
    assert records["tensor:2"].pp_mb_idx == 0
    assert records["tensor:2"].seq_idx == 1
    assert records["tensor:5"].pp_mb_idx == 1
    assert records["tensor:5"].seq_idx == 3

    prefetch = plan.to_summary_dict()["checkpoint_prefetch"]["part0:layers.0"]
    assert prefetch["prefetch_logical_bytes_per_instance"] == 80
    assert prefetch["instance_count"] == 2
    assert prefetch["prefetch_logical_bytes_total"] == 160
    assert prefetch["microbatches"] == [0, 1]


def test_checkpoint_prefetch_does_not_treat_external_context_as_internal_saved():
    context = TensorRef(
        tensor_id=1,
        name="index_table",
        shape=(1024, 6),
        dtype="int64",
        device="meta",
        num_bytes=1024 * 6 * 8,
        requires_grad=False,
    )
    output = tref(2, 64)
    plan = estimate_static_memory(
        [
            event(
                0,
                10,
                "aten.index.Tensor",
                inputs=[context],
                outputs=[output],
                module_path="layers.0._checkpoint_wrapped_module.router",
            ),
            event(
                5,
                20,
                "aten.index.Tensor",
                inputs=[context],
                phase="backward",
                execution_kind="recompute",
                module_path="layers.0._checkpoint_wrapped_module.router",
            ),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=0,
                inputs=(context,),
                outputs=(output,),
            )
        ],
        offload_ac_saved_tensors=True,
    )

    external = next(
        item for item in plan.tensor_lifetimes if item.tensor_id == "external:1"
    )
    assert external.kind == "external_input"
    assert external.resident_num_bytes == context.num_bytes
    summary = plan.to_summary_dict()
    assert summary["checkpoint_recompute_saved_tensor_count"] == 0
    assert summary["checkpoint_prefetch_unattributed_tensor_count"] == 0
    assert "part0:layers.0" not in summary["checkpoint_prefetch"]


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


def test_checkpoint_boundary_can_offload_saved_activation_but_keeps_metadata():
    x, saved, output, grad = tref(1, 32), tref(2, 64), tref(3, 16), tref(4, 64)
    plan = estimate_static_memory(
        [
            event(0, 10, "aten.relu.default", inputs=[x], outputs=[saved]),
            event(1, 11, "aten.relu.default", inputs=[saved], outputs=[output]),
            event(
                5,
                20,
                "aten.relu_backward.default",
                inputs=[saved],
                outputs=[grad],
                phase="backward",
            ),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.1",
                seq_idx=1,
                inputs=(saved,),
                outputs=(output,),
            )
        ],
        offload_ac_saved_tensors=True,
    )

    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert lifetime.kind == "checkpoint_saved_activation"
    assert lifetime.num_bytes == 64
    assert lifetime.modeled_num_bytes == 0
    assert lifetime.residency_policy == "offloaded"
    assert not any(item.tensor_id == "tensor:2" for item in plan.timeline_events)

    saved_record = next(
        item
        for item in plan.checkpoint_tensors
        if item.checkpoint_id == "part0:layers.1" and item.role == "input"
    )
    assert saved_record.shape == saved.shape
    assert saved_record.dtype == "float32"
    assert saved_record.num_bytes == 64
    assert saved_record.modeled_num_bytes == 0
    summary = plan.to_summary_dict()
    assert summary["checkpoint_tensor_logical_bytes"] == 64
    assert summary["checkpoint_tensor_modeled_bytes"] == 0
    marker = summary["checkpoint_saved_activations"]["part0:layers.1"]
    assert marker["marker_kind"] == "module"
    assert marker["logical_bytes_per_instance"] == 64
    assert marker["modeled_bytes_per_instance"] == 0
    assert marker["instance_count"] == 1
    assert marker["size_variants"] == [
        {
            "logical_bytes": 64,
            "modeled_bytes": 0,
            "tensor_count": 1,
            "instance_count": 1,
        }
    ]


def test_checkpoint_boundary_is_metadata_only_when_offload_is_disabled():
    x, saved, grad = tref(1), tref(2, 64), tref(3, 64)
    plan = estimate_static_memory(
        [
            event(0, 10, "aten.relu.default", inputs=[x], outputs=[saved]),
            event(5, 20, "aten.relu_backward.default", inputs=[saved], outputs=[grad], phase="backward"),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=0,
                inputs=(saved,),
                outputs=(),
            )
        ],
    )

    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert lifetime.kind == "checkpoint_saved_activation"
    assert lifetime.modeled_num_bytes is None
    assert lifetime.resident_num_bytes == 64
    assert any(item.tensor_id == "tensor:2" for item in plan.timeline_events)


def test_checkpoint_offload_excludes_non_gradient_context_inputs():
    saved = tref(2, 64)
    context = TensorRef(
        tensor_id=3,
        name="freqs",
        shape=(8, 8),
        dtype="float32",
        device="meta",
        num_bytes=256,
        requires_grad=False,
    )
    plan = estimate_static_memory(
        [
            event(
                0,
                10,
                "aten.relu.default",
                inputs=[tref(1), context],
                outputs=[saved],
            ),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=0,
                inputs=(saved, context),
                outputs=(),
            )
        ],
        offload_ac_saved_tensors=True,
    )

    records = {
        item.tensor_id: item
        for item in plan.checkpoint_tensors
        if item.role == "input"
    }
    assert records["tensor:2"].is_saved_activation is True
    assert records["tensor:2"].modeled_num_bytes == 0
    assert records["external:3"].is_saved_activation is False
    assert records["external:3"].modeled_num_bytes == 256
    assert records["external:3"].residency_policy == "resident"


def test_checkpoint_summary_preserves_size_variants_for_one_marker():
    small, large = tref(2, 64), tref(3, 128)
    plan = estimate_static_memory(
        [
            event(0, 10, "aten.relu.default", inputs=[tref(1)], outputs=[small]),
            event(1, 11, "aten.relu.default", inputs=[small], outputs=[large]),
        ],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:shared_block",
                seq_idx=0,
                inputs=(small,),
                outputs=(),
            ),
            CheckpointBoundaryEvent(
                checkpoint_id="part0:shared_block",
                seq_idx=1,
                inputs=(large,),
                outputs=(),
            ),
        ],
    )

    marker = plan.to_summary_dict()["checkpoint_saved_activations"][
        "part0:shared_block"
    ]
    assert marker["logical_bytes_per_instance"] is None
    assert marker["modeled_bytes_per_instance"] is None
    assert marker["instance_count"] == 2
    assert marker["logical_bytes_total"] == 192
    assert [item["logical_bytes"] for item in marker["size_variants"]] == [
        64,
        128,
    ]
    prefetch = plan.to_summary_dict()["checkpoint_prefetch"][
        "part0:shared_block"
    ]
    assert prefetch["prefetch_logical_bytes_per_instance"] is None
    assert prefetch["modeled_bytes_per_instance"] is None
    assert prefetch["instance_count"] == 2
    assert prefetch["prefetch_logical_bytes_total"] == 192
    assert [
        item["prefetch_logical_bytes"] for item in prefetch["size_variants"]
    ] == [64, 128]


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


def test_phase_peaks_include_memory_live_on_phase_entry():
    x, activation, grad = tref(1), tref(2), tref(3)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", inputs=[x], outputs=[activation], phase="forward"),
        event(5, 20, "aten.relu_backward.default", inputs=[activation], outputs=[grad], phase="backward"),
        event(9, 30, "optimizer.step", inputs=[grad], phase="optimizer"),
    ])

    assert plan.forward_peak_active_bytes == 32
    assert plan.backward_peak_active_bytes == 32
    assert plan.optimizer_peak_active_bytes == 16


def test_model_peak_excludes_optimizer_allocations():
    activation = tref(1, 32)
    grad = tref(2, 16)
    optimizer_state = tref(3, 512)
    plan = estimate_static_memory([
        event(0, 10, "aten.relu.default", outputs=[activation], phase="forward"),
        event(
            5,
            20,
            "aten.relu_backward.default",
            inputs=[activation],
            outputs=[grad],
            phase="backward",
        ),
        event(
            9,
            30,
            "optimizer.step",
            inputs=[grad],
            outputs=[optimizer_state],
            phase="optimizer",
        ),
    ])

    assert plan.model_active_bytes_peak == 48
    assert plan.peak_active_bytes == 528
    assert plan.to_summary_dict()["model_active_bytes_peak"] == 48


def test_missing_parameter_gradients_are_synthesized_through_optimizer():
    model = nn.Linear(4, 2, bias=False, device="meta")
    plan = estimate_static_memory(
        [
            event(0, 10, "aten.relu_backward.default", phase="backward"),
            event(5, 20, "optimizer.step", phase="optimizer"),
        ],
        model_parts=[model],
    )

    gradient = next(item for item in plan.tensor_lifetimes if item.tensor_id == "synthetic_grad:0:weight")
    assert gradient.kind == "gradient_accumulator"
    assert gradient.num_bytes == 4 * 2 * 4
    assert (gradient.birth_seq, gradient.death_seq) == (0, 5)
    assert "synthesized 1 missing parameter gradients" in " ".join(plan.notes)


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


def test_dtensor_local_parameter_materialization_is_not_counted_as_external_input():
    class ParameterModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1, 4, 8, device="meta"))

    model = ParameterModule()
    local_parameter = TensorRef(
        tensor_id=99,
        name="local_parameter",
        shape=(1, 4, 8),
        dtype="float32",
        device="meta",
        num_bytes=1 * 4 * 8 * 4,
    )
    plan = estimate_static_memory(
        [event(0, 10, "aten._to_copy.default", inputs=[local_parameter], outputs=[tref(1)])],
        model_parts=[model],
    )

    assert not any(item.tensor_id == "external:99" for item in plan.tensor_lifetimes)


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
        [
            event(
                0,
                10,
                "comm.allgather",
                inputs=[shard],
                outputs=[full],
                op_type="allgather",
            ),
            event(
                5,
                20,
                "aten.mm.default",
                inputs=[full],
                phase="backward",
                execution_kind="backward",
                module_path="layers.0.attention",
            ),
        ],
        comm_events=[FakeComm(op_id=10, comm_primitive="allgather", comm_dim="fsdp")],
        offload_ac_saved_tensors=True,
    )
    full_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    assert full_lifetime.kind == "fsdp_full_param"
    assert full_lifetime.num_bytes == 128
    assert full_lifetime.resident_num_bytes == 128
    assert plan.activation_offload_tensors == []
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


def test_fsdp_explicit_markers_remove_identity_lost_unsharded_parameter_alias():
    class ParameterModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1, 4, 8, device="meta"))

    full_param = TensorRef(
        tensor_id=99,
        name="full_parameter",
        shape=(2, 4, 8),
        dtype="bfloat16",
        device="meta",
        num_bytes=2 * 4 * 8 * 2,
    )
    plan = estimate_static_memory(
        [event(3, 20, "aten.mm.default", inputs=[full_param], outputs=[tref(1)])],
        model_parts=[ParameterModule()],
        fsdp_residency_events=[
            FSDPResidencyEvent("layer0", "alloc", 2, "forward", full_param.num_bytes, (123,)),
            FSDPResidencyEvent("layer0", "free", 5, "forward", full_param.num_bytes, (123,)),
        ],
    )

    assert not any(item.tensor_id == "external:99" for item in plan.tensor_lifetimes)
    residency = next(item for item in plan.tensor_lifetimes if item.kind == "fsdp_full_param")
    assert (residency.birth_seq, residency.death_seq) == (2, 5)
    assert "1 unsharded parameter aliases removed" in " ".join(plan.notes)


def test_parameter_bytes_are_persistent_and_counted():
    model = nn.Linear(4, 2, bias=False, device="meta")
    plan = estimate_static_memory([], model_parts=[model])
    assert plan.persistent_param_bytes == 4 * 2 * 4
    assert plan.peak_active_bytes == plan.persistent_param_bytes


def test_parameter_storage_dtype_changes_only_modeled_residency():
    model = nn.Linear(4, 2, bias=False, device="meta", dtype=torch.float32)
    plan = estimate_static_memory(
        [],
        model_parts=[model],
        parameter_storage_dtype="bfloat16",
    )

    parameter = next(
        item for item in plan.tensor_lifetimes if item.kind == "parameter_shard"
    )
    assert parameter.dtype == "float32"
    assert parameter.num_bytes == 4 * 2 * 4
    assert parameter.modeled_dtype == "bfloat16"
    assert parameter.modeled_num_bytes == 4 * 2 * 2
    assert parameter.residency_policy == "dtype_override"
    assert plan.persistent_param_bytes == 4 * 2 * 2
    assert plan.peak_active_bytes == plan.persistent_param_bytes


def test_parameter_storage_dtype_rejects_unknown_dtype():
    model = nn.Linear(4, 2, bias=False, device="meta")
    with pytest.raises(ValueError, match="Unsupported simulator memory dtype"):
        estimate_static_memory(
            [],
            model_parts=[model],
            parameter_storage_dtype="fp4",
        )


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
        event(
            5,
            20,
            "aten.relu.default",
            inputs=[b],
            outputs=[grad],
            phase="backward",
            execution_kind="recompute",
        ),
    ])

    trace = memory_plan_to_chrome_trace(plan)
    events = trace["traceEvents"]
    assert trace["displayTimeUnit"] == "ms"
    assert any(item["ph"] == "C" and item["name"] == "active_bytes" for item in events)
    assert any(item["ph"] == "X" and item["name"] == "forward" for item in events)
    assert any(item["ph"] == "X" and item["name"] == "backward" for item in events)
    assert any(item["ph"] == "X" and item["name"] == "recompute" for item in events)
    assert any(item["ph"] == "i" and item["name"] == "peak active bytes" for item in events)
    assert trace["metadata"]["forward_active_bytes_peak"] == 64
    assert trace["metadata"]["backward_active_bytes_peak"] == 64
    assert trace["metadata"]["optimizer_active_bytes_peak"] == 0
    assert trace["metadata"]["model_active_bytes_peak"] == 64

    export_memory_plan(plan, str(tmp_path))
    memory_dir = tmp_path / "memory"
    assert (memory_dir / "memory_trace.json").is_file()
    assert not (tmp_path / "memory_trace.json").exists()
    memory_events_header = (memory_dir / "memory_events.csv").read_text().splitlines()[0]
    memory_timeline_header = (memory_dir / "memory_timeline.csv").read_text().splitlines()[0]
    assert memory_events_header.startswith("event_id,seq_idx,phase,execution_kind,op_id")
    assert "execution_kind" in memory_events_header
    assert memory_timeline_header.startswith("seq_idx,phase,op_id,action")


def test_memory_summary_export_does_not_write_detailed_artifacts(tmp_path):
    plan = estimate_static_memory([], model_parts=[nn.Linear(4, 2, device="meta")])

    export_memory_summary(plan, str(tmp_path))

    memory_dir = tmp_path / "memory"
    assert {path.name for path in memory_dir.iterdir()} == {
        "memory_summary.json"
    }


def test_memory_plan_exports_checkpoint_tensor_metadata(tmp_path):
    saved = tref(2, 64)
    plan = estimate_static_memory(
        [event(0, 10, "aten.relu.default", inputs=[tref(1)], outputs=[saved])],
        checkpoint_boundary_events=[
            CheckpointBoundaryEvent(
                checkpoint_id="part0:layers.0",
                seq_idx=0,
                inputs=(saved,),
                outputs=(),
            )
        ],
        offload_ac_saved_tensors=True,
    )

    export_memory_plan(plan, str(tmp_path))
    checkpoint_csv = tmp_path / "memory" / "checkpoint_tensors.csv"
    assert checkpoint_csv.is_file()
    header, row = checkpoint_csv.read_text().splitlines()
    assert header.startswith(
        "checkpoint_id,marker_kind,seq_idx,tensor_id,role,shape,dtype"
    )
    assert "part0:layers.0" in row
    assert "offloaded" in row


def test_memory_plan_exports_none_mode_activation_offload_metadata(tmp_path):
    saved = tref(2, 64)
    plan = estimate_static_memory(
        [
            event(
                0,
                10,
                "aten.relu.default",
                inputs=[tref(1)],
                outputs=[saved],
                module_path="layers.0",
            ),
            event(
                5,
                20,
                "aten.relu.default",
                inputs=[saved],
                phase="backward",
                execution_kind="backward",
                module_path="layers.0",
            ),
        ],
        offload_ac_saved_tensors=True,
    )

    export_memory_plan(plan, str(tmp_path))
    activation_csv = (
        tmp_path / "memory" / "activation_offload_tensors.csv"
    )
    assert activation_csv.is_file()
    header, row = activation_csv.read_text().splitlines()
    assert header.startswith(
        "checkpoint_id,marker_kind,seq_idx,tensor_id,role,shape,dtype"
    )
    assert "part0:layers.0" in row
    assert "activation_saved" in row
    assert "offloaded" in row


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
