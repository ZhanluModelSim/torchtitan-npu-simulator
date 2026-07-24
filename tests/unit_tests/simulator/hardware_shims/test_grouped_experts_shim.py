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

    capture = OpDispatchCapture()
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
        output.backward(grad_output)

    assert x.grad is not None and x.grad.shape == x.shape
    assert w13.grad is not None and w13.grad.shape == w13.shape
    assert w2.grad is not None and w2.grad.shape == w2.shape
    assert scores.grad is not None and scores.grad.shape == scores.shape
    # The shim only repairs autograd connectivity. Shape-only gradients must
    # not add fabricated operators to the downstream workload graph.
    backward_op_types = {
        event.raw_op_type for event in capture._events[forward_event_count:]
    }
    assert backward_op_types <= {"aten.detach.default"}
