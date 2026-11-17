# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the repository-owned GDN custom-op boundary.

The Triton kernel itself requires an NPU, but its input validation and
normalization are device-independent.  Keeping these checks on CPU catches
shape/dtype/reset regressions before an accelerator job is started.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[5] / "torchtitan_npu" / "ops" / "triton" / "gdn" / "gated_delta.py"


@pytest.fixture(scope="module")
def gdn():
    spec = importlib.util.spec_from_file_location("gdn_validation_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(*, dtype=torch.bfloat16, tokens=64, reset=None):
    shape = (1, tokens, 1, 64)
    q = torch.randn(shape, dtype=dtype)
    k = torch.randn(shape, dtype=dtype)
    v = torch.randn(shape, dtype=dtype)
    g = torch.randn(1, tokens, 1, dtype=torch.float32)
    beta = torch.randn(1, tokens, 1, dtype=torch.float32)
    if reset is None:
        reset = torch.zeros(1, tokens, dtype=torch.bool)
    return q, k, v, g, beta, reset


def test_gdn_l2_normalize_preserves_dtype_and_normalizes(gdn):
    values = torch.tensor([[[[3.0, 4.0] + [0.0] * 62]]])
    normalized, inverse = gdn._l2_normalize(values)
    expected_inverse = torch.rsqrt(values.square().sum(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(inverse, expected_inverse)
    torch.testing.assert_close(normalized.float().norm(dim=-1), torch.ones(1, 1, 1), atol=1e-6, rtol=1e-6)
    assert normalized.dtype == values.dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gdn_l2_normalize_matches_float64_oracle(gdn, dtype):
    """The CPU normalization seam must agree with an independent high-precision oracle."""
    values = torch.tensor(
        [[[[0.25, -1.5, 2.0, 3.25], [1.0, 0.5, -0.75, 2.5]]]],
        dtype=dtype,
    )
    normalized, inverse = gdn._l2_normalize(values)

    values64 = values.to(torch.float64)
    expected_inverse = torch.rsqrt(values64.square().sum(dim=-1, keepdim=True) + 1e-6)
    expected_normalized = (values64 * expected_inverse).to(dtype)

    torch.testing.assert_close(inverse, expected_inverse.float(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(normalized, expected_normalized, atol=2e-3, rtol=2e-3)


def test_gdn_l2_normalize_cpu_gradient_matches_oracle(gdn):
    """Normalization remains differentiable on CPU, protecting q/k math before NPU execution."""
    values = torch.tensor(
        [[[[0.25, -1.5, 2.0, 3.25], [1.0, 0.5, -0.75, 2.5]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    upstream = torch.tensor(
        [[[[0.5, -0.25, 1.25, -0.75], [1.0, -0.5, 0.25, 0.75]]]],
        dtype=torch.float32,
    )

    normalized, inverse = gdn._l2_normalize(values)
    loss = (normalized * upstream).sum() + 0.1 * inverse.sum()
    (actual_grad,) = torch.autograd.grad(loss, values)

    values64 = values.detach().to(torch.float64).requires_grad_(True)
    inverse64 = torch.rsqrt(values64.square().sum(dim=-1, keepdim=True) + 1e-6)
    normalized64 = values64 * inverse64
    loss64 = (normalized64 * upstream.to(torch.float64)).sum() + 0.1 * inverse64.sum()
    (expected_grad,) = torch.autograd.grad(loss64, values64)

    torch.testing.assert_close(actual_grad, expected_grad.float(), atol=2e-5, rtol=2e-5)


def test_gdn_validation_accepts_supported_inputs(gdn):
    gdn._validate_inputs(_inputs(), scale=0.5)


def test_gdn_validation_accepts_float16_and_max_batch_head_product(gdn):
    gdn._validate_inputs(_inputs(dtype=torch.float16), scale=0.5)
    shape = (8191, 64, 1, 64)
    q = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    g = torch.empty(shape[:3], dtype=torch.float32, device="meta")
    reset = torch.empty(shape[:2], dtype=torch.bool, device="meta")
    gdn._validate_inputs((q, q, q, g, g, reset), scale=0.5)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param({"tokens": 63}, ValueError, "positive multiple of 64", id="token-length"),
        pytest.param({"dtype": torch.float32}, TypeError, "bfloat16 or float16", id="q-dtype"),
        pytest.param({"reset": torch.zeros(1, 64)}, ValueError, "reset", id="reset-shape"),
    ],
)
def test_gdn_validation_rejects_invalid_inputs(gdn, kwargs, error, message):
    with pytest.raises(error, match=message):
        gdn._validate_inputs(_inputs(**kwargs), scale=0.5)


def test_gdn_validation_rejects_nonfinite_scale(gdn):
    with pytest.raises(ValueError, match="scale must be finite"):
        gdn._validate_inputs(_inputs(), scale=float("inf"))
