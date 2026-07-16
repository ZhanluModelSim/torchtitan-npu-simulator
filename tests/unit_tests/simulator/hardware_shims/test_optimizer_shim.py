# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.optimizer_shim import _meta_safe_fused_adamw


def test_meta_safe_fused_adamw_does_not_read_values_or_dispatch_updates():
    param = torch.empty(4, device="meta")
    grad = torch.empty_like(param)
    exp_avg = torch.empty_like(param)
    exp_avg_sq = torch.empty_like(param)
    state_step = torch.empty((), device="meta")

    capture = OpDispatchCapture()
    with capture:
        _meta_safe_fused_adamw(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [],
            [state_step],
            amsgrad=False,
            beta1=0.9,
            beta2=0.999,
            lr=1e-3,
            weight_decay=0.01,
            eps=1e-8,
            maximize=False,
        )

    event = next(
        event for event in capture.memory_events()
        if event.raw_op_type == "npu.npu_apply_adam_w.default"
    )
    assert len(event.inputs) == 5
    assert len(event.outputs) == 4
    assert {ref.tensor_id for ref in event.outputs} <= {
        ref.tensor_id for ref in event.inputs
    }
