# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torchtitan_simulator.runtime import meta_runtime


def test_float32_router_projection_retains_cast_gradients_against_cpu_reference():
    inputs = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True)
    weight = torch.tensor([[2, -1]], dtype=torch.bfloat16, requires_grad=True)

    with meta_runtime(), torch.autocast("meta", dtype=torch.float32):
        output = F.linear(inputs, weight)
        output.square().sum().backward()

    assert output.dtype == torch.float32
    assert torch.equal(output, torch.tensor([[0.0], [2.0]]))
    assert torch.equal(inputs.grad, torch.tensor([[0, 0], [8, -4]], dtype=torch.bfloat16))
    assert torch.equal(weight.grad, torch.tensor([[12, 16]], dtype=torch.bfloat16))
