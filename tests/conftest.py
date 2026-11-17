# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Shared pytest fixtures for the smoke tests under tests/smoke_tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def npu_available():
    """Return whether a real NPU runtime is available."""
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture(scope="session")
def npu_device(npu_available):
    """Provide a shared NPU device fixture for smoke tests."""
    if not npu_available:
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def stable_randn(*shape, device, dtype=torch.float32, scale=0.01, requires_grad=False):
    """Generate small-amplitude random tensors to avoid unstable smoke inputs."""
    tensor = torch.randn(*shape, dtype=torch.float32, device=device) * scale
    tensor = tensor.to(dtype)
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def assert_tensor_finite(value, message="Tensor should be finite"):
    """Check finiteness on CPU to avoid NPU-side isfinite inconsistencies."""
    if not torch.isfinite(value.detach().float().cpu()).all().item():
        raise AssertionError(message)
