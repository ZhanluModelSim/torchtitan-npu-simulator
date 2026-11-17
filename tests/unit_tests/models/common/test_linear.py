# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.models.common.linear import SharedLinear


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_shared_linear_detaches_weights_and_restores_mode(compiled):
    with torch.random.fork_rng():
        head = SharedLinear.Config(in_features=4, out_features=3, bias=True).build()
        projection = torch.compile(head, backend="aot_eager", fullgraph=True) if compiled else head
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
        expected = torch.nn.functional.linear(inputs, head.weight, head.bias).detach()
        with head.detached_weight():
            actual = projection(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
        assert head.weight.grad is None and head.bias.grad is None
        torch.testing.assert_close(inputs.grad, head.weight.sum(0).expand_as(inputs))

        projection(inputs).sum().backward()
        torch.testing.assert_close(head.weight.grad, inputs.sum(0).expand_as(head.weight))
        torch.testing.assert_close(head.bias.grad, torch.full_like(head.bias, inputs.shape[0]))


def test_shared_linear_restores_nested_mode_after_exception():
    with torch.random.fork_rng():
        head = SharedLinear.Config(in_features=4, out_features=3).build()
        with pytest.raises(RuntimeError, match="abort"), head.detached_weight():
            with head.detached_weight():
                pass
            assert head._detach_weight
            raise RuntimeError("abort")
        assert not head._detach_weight
