# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.simulator import meta_env
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.saved_tensors import AutogradSavedTensorCapture


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_reverse_backward_order_captures_the_forward_source_microbatch(monkeypatch, device):
    context = {"stage": 0, "mb_idx": 0, "comp_type": "F", "phase": "forward"}
    monkeypatch.setattr(meta_env, "_pp_context", context)
    capture = OpDispatchCapture(phase_provider=lambda: context["phase"])
    losses = []
    with capture, AutogradSavedTensorCapture():
        for mb in (0, 1):
            context.update(mb_idx=mb, comp_type="F", phase="forward")
            capture.begin_chunk((0, "F"))
            x = torch.ones(4, device=device, requires_grad=True)
            losses.append(x.sin().cos().sum())
            capture.end_chunk()
        for mb in (1, 0):
            context.update(mb_idx=mb, comp_type="B", phase="backward")
            capture.begin_chunk((0, "B"))
            losses[mb].backward()
            capture.end_chunk()
    events = capture.memory_events()
    assert {event.pp_mb_idx for event in events} == {0}
    assert {event.comp_type for event in events} == {"F", "B"}
    assert capture.class_instance_counts == {(0, "F"): 2, (0, "B"): 2}
    assert all(saved.pp_mb_idx == 0 for saved in capture.autograd_saved_tensor_events())


@pytest.mark.parametrize("schedule_name", ["ScheduleGPipe", "Schedule1F1B"])
def test_actual_single_stage_pipeline_schedule_keeps_paired_source(tmp_path, monkeypatch, schedule_name):
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage, schedules

    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    monkeypatch.setattr(meta_env, "_pp_context", {"stage": -1, "mb_idx": -1, "comp_type": "", "phase": "forward"})
    try:
        meta_env.patch_device_type_to_meta()
        model = torch.nn.Linear(4, 4, device="cpu")
        stage = PipelineStage(
            model,
            0,
            1,
            torch.device("cpu"),
            input_args=torch.ones(2, 4, device="cpu"),
            output_args=torch.ones(2, 4, device="cpu"),
        )
        schedule = getattr(schedules, schedule_name)(
            stage, n_microbatches=2, loss_fn=lambda output, target: (output - target).square().sum()
        )
        capture = OpDispatchCapture(phase_provider=lambda: meta_env._pp_context["phase"])
        with capture, AutogradSavedTensorCapture():
            schedule.step(torch.ones(4, 4, device="cpu"), target=torch.ones(4, 4, device="cpu"))
        capture.finalize_autograd_saved_tensors()
        events = capture.memory_events()
        forward_math = [event for event in events if event.raw_op_type == "aten.addmm.default"]
        backward_math = [event for event in events if event.raw_op_type == "aten.mm.default"]
        assert forward_math and backward_math
        assert {event.pp_mb_idx for event in (*forward_math, *backward_math)} == {0}
        assert {(event.pp_stage, event.comp_type) for event in events} >= {(0, "F"), (0, "B")}
        assert capture.class_instance_counts[(0, "F")] == 2
        assert capture.class_instance_counts[(0, "B")] == 2
    finally:
        meta_env.unpatch_device_type_to_meta()
        dist.destroy_process_group()
