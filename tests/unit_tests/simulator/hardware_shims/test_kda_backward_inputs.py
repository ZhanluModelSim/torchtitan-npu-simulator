# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.kda_shim import _SimChunkKDAFn


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_kda_backward_records_saved_gate_parameters(device):
    values = [torch.ones(1, 8, 2, 4, device=device, requires_grad=True) for _ in range(5)]
    parameters = [
        torch.ones(2, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    ]
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0])
    with capture:
        result = _SimChunkKDAFn.apply(*values, *parameters, "layers.0.attention")
        phase[0] = "backward"
        result.sum().backward()
    backward = next(event for event in capture._events if event.raw_op_type == "triton_ascend_kernels.chunk_kda_grad")
    assert len(backward.inputs) == 8
    assert [item.shape for item in backward.inputs[-3:-1]] == [tuple(p.shape) for p in parameters]
    assert [item.dtype for item in backward.inputs[-3:-1]] == ["float32", "float32"]
    memory_event = next(
        event for event in capture.memory_events() if event.raw_op_type == "triton_ascend_kernels.chunk_kda_grad"
    )
    assert [ref.tensor_id for ref in memory_event.inputs[-3:-1]] == [capture.tensor_id(p) for p in parameters]
    assert len(backward.outputs) == 7
    assert all(value.grad is not None and value.grad.shape == value.shape for value in [*values, *parameters])
