# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.models.common.rope import HalfRotation
from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE


@pytest.mark.parametrize("seam,inverse", [("interleaved", True), ("half", False)])
def test_quarter_turn_and_gradient_on_partial_noncontiguous_input(seam, inverse):
    dtype = torch.bfloat16
    if seam == "interleaved":

        def apply(x, cos, sin, *, inverse=False):
            return WorkaroundComplexRoPE.apply_rotary_emb(x, None, (cos, sin), inverse=inverse)

        # All-zero cos / all-one sin (pair-duplicated by construction): a
        # quarter turn whose direction flips with the batch row.
        cos = torch.zeros(2, 3, 1, 4)
        sin = torch.ones(2, 3, 1, 4)
        sin[1] = -1
        real_ids, imag_ids = [0, 2], [1, 3]
    else:
        apply = HalfRotation.Config().build()
        cos = torch.zeros(2, 3, 1, 2)
        sin = torch.ones(2, 3, 1, 2)
        sin[1] = -1
        real_ids, imag_ids = [0, 1], [2, 3]

    full = torch.arange(48, dtype=dtype).reshape(2, 3, 1, 8).requires_grad_()
    x = full[..., 4:]
    assert not x.is_contiguous()
    actual = apply(x, cos, sin, inverse=inverse)
    expected = torch.empty_like(x)
    sign = torch.tensor([1, -1], dtype=dtype).view(2, 1, 1, 1) * (-1 if inverse else 1)
    expected[..., real_ids] = -x[..., imag_ids] * sign
    expected[..., imag_ids] = x[..., real_ids] * sign
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.arange(1, 25, dtype=dtype).reshape_as(x)
    actual.backward(grad)
    expected_grad = torch.zeros_like(full)
    expected_grad[..., [4 + i for i in real_ids]] = grad[..., imag_ids] * sign
    expected_grad[..., [4 + i for i in imag_ids]] = -grad[..., real_ids] * sign
    torch.testing.assert_close(full.grad, expected_grad, rtol=0, atol=0)
