# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_npu.simulator.hardware_shims.grouped_experts_shim import (
    run_meta_grouped_experts,
)
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory


def test_meta_grouped_experts_produces_shape_correct_expert_gradients():
    w13 = torch.empty(4, 16, 8, device="meta", requires_grad=True)
    w2 = torch.empty(4, 8, 8, device="meta", requires_grad=True)
    x = torch.empty(12, 8, device="meta", requires_grad=True)
    counts = torch.empty(4, dtype=torch.int32, device="meta")
    scores = torch.empty(12, 1, device="meta", requires_grad=True)

    forward_saved_tensors: list[torch.Tensor] = []

    def fake_forward(w13, w2, _w3, x, counts, limit, scores):
        assert counts.shape == (4,)
        assert limit == 7.0
        assert scores.shape == (12, 1)
        pre_activation = torch.empty(x.shape[0], w13.shape[-2], device=x.device)
        activated_hidden = torch.empty(x.shape[0], w2.shape[1], device=x.device)
        scaled_hidden = torch.empty_like(activated_hidden)
        output = torch.empty(x.shape[0], w2.shape[1], device=x.device)
        forward_saved_tensors.extend([pre_activation, activated_hidden, scaled_hidden])
        return output, (pre_activation, activated_hidden, scaled_hidden)

    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0])
    grad_output = torch.empty(12, 8, device="meta")
    with capture, AutogradSavedTensorCapture():
        output = run_meta_grouped_experts(
            fake_forward,
            w13,
            w2,
            x,
            counts,
            7.0,
            scores,
        )
        forward_event_count = len(capture._events)
        phase[0] = "backward"
        output.backward(grad_output)
    capture.finalize_autograd_saved_tensors()

    assert x.grad is not None and x.grad.shape == x.shape
    assert w13.grad is not None and w13.grad.shape == w13.shape
    assert w2.grad is not None and w2.grad.shape == w2.shape
    assert scores.grad is not None and scores.grad.shape == scores.shape
    backward_events = capture._events[forward_event_count:]
    grouped_mm_events = [event for event in backward_events if event.raw_op_type == "aten._grouped_mm.default"]
    assert len(grouped_mm_events) == 4
    assert [event.outputs[0].shape for event in grouped_mm_events] == [
        (12, 8),
        (4, 8, 8),
        (12, 8),
        (4, 16, 8),
    ]
    assert all(event.phase == "backward" for event in grouped_mm_events)

    mul_events = [event for event in backward_events if event.raw_op_type == "aten.mul.Tensor"]
    swiglu_event = next(
        event
        for event in backward_events
        if event.raw_op_type == "npu.npu_swiglu_backward.default"
    )
    assert len(mul_events) == 2
    assert grouped_mm_events[0].op_id in mul_events[0].predecessors
    assert grouped_mm_events[0].op_id in mul_events[1].predecessors
    assert mul_events[0].op_id in swiglu_event.predecessors
    assert swiglu_event.op_id in grouped_mm_events[2].predecessors
    assert swiglu_event.op_id in grouped_mm_events[3].predecessors

    # The shim must save the identities produced by its forward graph.  This
    # keeps the producer -> saved-slot -> lifetime chain intact for both peak
    # memory and activation-offload accounting.
    saved_ids = {item.tensor_id for item in capture.autograd_saved_tensor_events()}
    forward_saved_ids = {capture.tensor_id(tensor) for tensor in forward_saved_tensors}
    assert forward_saved_ids <= saved_ids
    forward_output_ids = {
        output.tensor_id
        for event in capture.memory_events()
        if event.phase == "forward"
        for output in event.outputs
    }
    assert forward_saved_ids <= forward_output_ids

    plan = estimate_static_memory(
        capture.memory_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
        offload_ac_saved_tensors=True,
    )
    lifetimes = {item.tensor_id: item for item in plan.tensor_lifetimes}
    assert all(
        lifetimes[f"tensor:{tensor_id}"].kind == "offloaded_activation"
        for tensor_id in forward_saved_ids
    )
