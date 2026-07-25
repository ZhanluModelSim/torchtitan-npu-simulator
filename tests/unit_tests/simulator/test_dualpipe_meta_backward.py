# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch

from torchtitan_npu.patches.torch.pipelining import backward_maybe_with_nosync
from torchtitan_npu.simulator import meta_env


def test_meta_split_backward_replays_weight_grad_from_tensor_outputs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(meta_env, "_is_meta_simulation", True)
    module = torch.nn.Linear(4, 3, device="meta")
    stage = SimpleNamespace(submod=module)
    stage_input = torch.empty(2, 4, device="meta", requires_grad=True)
    stage_output = module(stage_input)
    output_grad = torch.empty_like(stage_output)

    input_grads, param_groups = backward_maybe_with_nosync(
        stage,
        "input",
        {
            "stage_output": [stage_output],
            "output_grads": [output_grad],
            "input_values": [stage_input],
        },
    )
    backward_maybe_with_nosync(
        stage,
        "weight",
        {"param_groups": param_groups},
    )

    assert input_grads[0] is not None
    assert all(param.grad is not None for param in module.parameters())
