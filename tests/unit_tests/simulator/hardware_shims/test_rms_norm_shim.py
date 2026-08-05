# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import SimRMSNorm


def test_sim_rms_norm_records_forward_and_backward_and_preserves_gradients():
    shim = SimRMSNorm(RMSNorm(RMSNorm.Config(normalized_shape=16, eps=1e-6)))
    x = torch.empty((2, 3, 16), device="meta", requires_grad=True)
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = shim(x)
        phase["value"] = "backward"
        output.sum().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert shim.weight.grad is not None
    assert shim.weight.grad.shape == shim.weight.shape

    nodes = capture.build_nodes().values()
    forward_names = {node.annotations["raw_op_type"] for node in nodes if node.annotations["phase"] == "forward"}
    backward_names = {node.annotations["raw_op_type"] for node in nodes if node.annotations["phase"] == "backward"}
    assert "npu.npu_rms_norm.default" in forward_names
    assert "npu.npu_rms_norm_backward.default" in backward_names
