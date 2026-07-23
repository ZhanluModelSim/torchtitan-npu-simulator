# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.models.common.rope import (
    apply_rotary_emb_complex as llama_apply_rotary_emb,
)
from torchtitan.models.common.rope import (
    apply_rotary_emb_cos_sin as qwen_apply_rotary_emb,
)
from torchtitan.models.common.rope import (
    apply_rotary_emb_single_complex as deepseek_apply_rotary_emb,
)

from tests.conftest import assert_tensor_finite, stable_randn
from torchtitan_npu.converters.kernels.rope import (
    npu_apply_rotary_emb_complex as npu_apply_rotary_emb_llama,
)
from torchtitan_npu.converters.kernels.rope import (
    npu_apply_rotary_emb_cos_sin as npu_apply_rotary_emb_qwen,
)
from torchtitan_npu.converters.kernels.rope import (
    npu_apply_rotary_emb_single_complex as npu_apply_rotary_emb_deepseek,
)

pytestmark = pytest.mark.smoke


def _complex_freqs(shape, device):
    real = stable_randn(*shape, dtype=torch.float32, device=device)
    imag = stable_randn(*shape, dtype=torch.float32, device=device)
    return torch.complex(real, imag)


def _assert_tensors_close(
    expected: torch.Tensor,
    actual: torch.Tensor,
    message_prefix: str,
    *,
    rtol: float,
    atol: float,
):
    assert torch.allclose(expected.float(), actual.float(), rtol=rtol, atol=atol), (
        f"{message_prefix}: max_diff={torch.max(torch.abs(expected.float() - actual.float())).item()}"
    )


def test_rope_deepseek(npu_device):
    x = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    freqs_cis = _complex_freqs((128, 32), npu_device)

    output = npu_apply_rotary_emb_deepseek(x, freqs_cis)

    assert output.shape == x.shape
    assert_tensor_finite(output)


def test_rope_llama(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    freqs_cis = _complex_freqs((128, 32), npu_device)

    q_out, k_out = npu_apply_rotary_emb_llama(xq, xk, freqs_cis)

    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_rope_qwen(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    rope_cache = stable_randn(128, 128, dtype=torch.float32, device=npu_device)

    q_out, k_out = npu_apply_rotary_emb_qwen(xq, xk, rope_cache)

    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_npu_apply_rotary_emb_llama_precision(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    freqs_cis = _complex_freqs((128, 32), npu_device)

    expected_q, expected_k = llama_apply_rotary_emb(xq, xk, freqs_cis)
    actual_q, actual_k = npu_apply_rotary_emb_llama(xq, xk, freqs_cis)

    assert expected_q.shape == actual_q.shape
    assert expected_k.shape == actual_k.shape
    _assert_tensors_close(expected_q, actual_q, "Query output mismatch", rtol=1e-5, atol=1e-5)
    _assert_tensors_close(expected_k, actual_k, "Key output mismatch", rtol=1e-5, atol=1e-5)


def test_npu_apply_rotary_emb_qwen_precision(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    rope_cache = stable_randn(128, 128, dtype=torch.float32, device=npu_device)

    expected_q, expected_k = qwen_apply_rotary_emb(xq, xk, rope_cache)
    actual_q, actual_k = npu_apply_rotary_emb_qwen(xq, xk, rope_cache)

    assert expected_q.shape == actual_q.shape
    assert expected_k.shape == actual_k.shape
    _assert_tensors_close(expected_q, actual_q, "Query output mismatch", rtol=1e-5, atol=1e-5)
    _assert_tensors_close(expected_k, actual_k, "Key output mismatch", rtol=1e-5, atol=1e-5)


def test_npu_apply_rotary_emb_deepseek_precision(npu_device):
    x = stable_randn(
        2,
        128,
        8,
        64,
        dtype=torch.float32,
        device=npu_device,
    )
    freqs_cis = _complex_freqs((128, 32), npu_device)

    expected = deepseek_apply_rotary_emb(x, freqs_cis)
    actual = npu_apply_rotary_emb_deepseek(x, freqs_cis)

    assert expected.shape == actual.shape

    _assert_tensors_close(expected, actual, "Output mismatch", rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("inverse", [False, True])
def test_npu_apply_rotary_emb_deepseek_stacked_cache_bitwise(npu_device, inverse):
    x_seed = stable_randn(2, 128, 8, 64, dtype=torch.bfloat16, device=npu_device)
    upstream = stable_randn(*x_seed.shape, dtype=torch.bfloat16, device=npu_device)
    positions = torch.arange(128, device=npu_device).unsqueeze(0)
    positions[0, -1] = -1
    freqs_cis = _complex_freqs((128, 32), npu_device)
    rope_cache = torch.view_as_real(freqs_cis).movedim(-1, 0).repeat_interleave(2, dim=-1)

    expected_x = x_seed.clone().requires_grad_(True)
    actual_x = x_seed.clone().requires_grad_(True)
    expected_freqs = freqs_cis.conj() if inverse else freqs_cis
    expected = npu_apply_rotary_emb_deepseek(expected_x, expected_freqs, positions)
    actual = npu_apply_rotary_emb_deepseek(
        actual_x,
        rope_cache,
        positions,
        inverse=inverse,
    )
    expected.backward(upstream)
    actual.backward(upstream)

    assert torch.equal(expected, actual)
    assert expected_x.grad is not None
    assert actual_x.grad is not None
    assert torch.equal(expected_x.grad, actual_x.grad)
