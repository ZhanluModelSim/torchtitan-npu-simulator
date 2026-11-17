# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression checks for the torch_npu mHC meta kernels under schema growth.

op-plugin appends optional arguments over time (``inner_precise``,
op-plugin#5836, grew ``npu_mhc_pre_backward`` from 13 to 14 arguments) and
torch_npu's autograd passes every schema argument positionally, so the fake
implementations must tolerate trailing extras.  These run on CPU tensors but
need the real torch_npu/CANN import boundary, so they live in the smoke tier
(the unit-test conftest fakes ``cann_ops_transformer``).
"""

import pytest
import torch

pytest.importorskip("torchtitan")
pytest.importorskip("torch_npu")
pytest.importorskip("cann_ops_transformer")
mhc_ops = pytest.importorskip("torchtitan_npu.ops.ascendc.mhc")


def test_fake_mhc_pre_backward_tolerates_appended_positional_args():
    x = torch.randn(2, 3)
    phi = torch.randn(2, 4)
    alpha = torch.randn(2)
    dummies = [torch.randn(2) for _ in range(7)]
    gamma = torch.randn(4)

    # 13-arg legacy layout, 14 = inner_precise (op-plugin#5836), 15 = a
    # further future append.
    for extra in ((), (1,), (1, 0)):
        outputs = mhc_ops._fake_mhc_pre_backward(x, phi, alpha, *dummies, gamma, 1e-6, None, *extra)
        assert len(outputs) == 5
        assert outputs[0].shape == x.shape
        assert outputs[3].shape == phi[:, 0].shape
        assert outputs[4].shape == gamma.shape

        outputs = mhc_ops._fake_mhc_pre_backward(x, phi, alpha, *dummies, None, 1e-6, None, *extra)
        assert outputs[4].numel() == 0


def test_fake_mhc_sinkhorn_backward_tolerates_appended_positional_args():
    grad_y = torch.randn(2, 3)

    for extra in ((), (1,)):
        outputs = mhc_ops._fake_mhc_sinkhorn_backward(grad_y, torch.randn(2), torch.randn(2), *extra)
        assert outputs.shape == grad_y.shape
