# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.grouped_experts_shim import (
    run_meta_grouped_experts,
)


def test_meta_grouped_experts_produces_shape_correct_expert_gradients():
    w13 = torch.empty(4, 16, 8, device="meta", requires_grad=True)
    w2 = torch.empty(4, 8, 8, device="meta", requires_grad=True)
    x = torch.empty(12, 8, device="meta", requires_grad=True)
    counts = torch.empty(4, dtype=torch.int32, device="meta")
    scores = torch.empty(12, 1, device="meta", requires_grad=True)

    def fake_forward(w13, w2, _w3, x, counts, limit, scores):
        assert counts.shape == (4,)
        assert limit == 7.0
        assert scores.shape == (12, 1)
        return torch.empty(x.shape[0], w2.shape[1], device=x.device)

    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0])
    grad_output = torch.empty(12, 8, device="meta")
    with capture:
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
