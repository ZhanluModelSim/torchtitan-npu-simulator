# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("offload", [False, True])
def test_ctx_attribute_retention_is_not_denied_by_saved_slots(device, offload):
    retained = []

    class Retained(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x.sin())
            ctx.attribute = x.cos()
            retained.append(ctx.attribute)
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            (saved,) = ctx.saved_tensors
            return grad * ctx.attribute + saved

    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0])
    x = torch.ones(4, device=device, requires_grad=True)
    with capture, AutogradSavedTensorCapture():
        loss = Retained.apply(x).sum()
        phase[0] = "backward"
        loss.backward()
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
        offload_ac_saved_tensors=offload,
    )
    retained_id = capture.tensor_id(retained[0])
    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{retained_id}")
    use = next(event for event in capture.memory_events() if event.raw_op_type == "aten.mul.Tensor")
    assert lifetime.kind == "activation"
    assert lifetime.death_seq >= use.seq_idx
    assert lifetime.resident_num_bytes == 16
    assert lifetime.reason == "backward_use_without_saved_slot"
