# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.moe_permutation_shim import (
    run_meta_moe_token_permute,
)


def test_meta_moe_token_permute_records_real_forward_and_backward_ops():
    tokens = torch.empty((8, 4), device="meta", requires_grad=True)
    indices = torch.empty((8, 2), dtype=torch.int64, device="meta")
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output, sorted_indices = run_meta_moe_token_permute(
            torch_npu.npu_moe_token_permute,
            tokens,
            indices,
        )
        phase["value"] = "backward"
        output.sum().backward()

    assert output.shape == (16, 4)
    assert sorted_indices.shape == (16,)
    assert tokens.grad is not None
    assert tokens.grad.shape == tokens.shape

    nodes = capture.build_nodes().values()
    names_by_phase = {
        phase_name: {node.annotations["raw_op_type"] for node in nodes if node.annotations["phase"] == phase_name}
        for phase_name in ("forward", "backward")
    }
    assert "npu.npu_moe_token_permute.default" in names_by_phase["forward"]
    assert "npu.npu_moe_token_permute_grad.default" in names_by_phase["backward"]
