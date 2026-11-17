# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical CPU contracts for the eager DeepSeek-V4 MHC composition layers.

The Triton and AscendC overrides run on NPU and are intentionally not imported
here. These tests exercise the production CPU implementation against an
independent torch reference so that parameter layout, Sinkhorn normalization,
and the pre/post/head equations stay covered without requiring an accelerator.
The same reference also pins the FSDP parameter-storage contract: FP32 storage
and BF16 storage holding equal values must compute identically, because the
layers promote with ``.float()`` themselves.
"""

import copy

import pytest
import torch

from torchtitan_npu.models.deepseek_v4.mhc import HcHead, HcPost, HcPre


def _set_hc_pre_parameters(module):
    """Use an asymmetric fixture that makes Sinkhorn epsilon placement observable."""
    with torch.no_grad():
        module.hc_fn.copy_(
            torch.tensor(
                [
                    [0.10, -0.20, 0.05, 0.00, 0.10, -0.05],
                    [-0.25, 0.15, 0.20, -0.10, 0.00, 0.12],
                    [0.05, 0.30, -0.15, 0.20, -0.20, 0.10],
                    [-0.20, 0.10, 0.25, 0.15, 0.05, -0.30],
                    [1.00, 0.00, -1.00, 0.00, 0.00, 0.00],
                    [0.00, 0.50, 0.00, -0.50, 0.00, 0.00],
                    [0.00, 0.00, 0.75, 0.00, -0.75, 0.00],
                    [0.00, 0.00, 0.00, 1.00, 0.00, -1.00],
                ],
                dtype=module.hc_fn.dtype,
            )
        )
        module.hc_base.copy_(
            torch.tensor(
                [0.10, -0.30, 0.20, 0.40, 0.20, -0.40, 0.80, -0.10],
                dtype=module.hc_base.dtype,
            )
        )
        module.hc_scale.copy_(torch.tensor([1.20, 0.80, 1.30], dtype=module.hc_scale.dtype))


def _set_hc_head_parameters(module):
    with torch.no_grad():
        module.hc_fn.copy_(
            torch.tensor(
                [
                    [0.30, -0.10, 0.20, 0.00, -0.25, 0.15],
                    [-0.20, 0.35, -0.05, 0.25, 0.10, -0.30],
                ],
                dtype=module.hc_fn.dtype,
            )
        )
        module.hc_base.copy_(torch.tensor([0.20, -0.35], dtype=module.hc_base.dtype))
        module.hc_scale.copy_(torch.tensor([0.90], dtype=module.hc_scale.dtype))


def _round_parameters_to_bf16(module):
    """Keep FP32 storage but make every value exactly representable in BF16."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(parameter.to(torch.bfloat16).to(torch.float32))


def _store_parameters_as_bf16(module):
    """Move parameter values into BF16 storage, as FSDP's default policy does."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(parameter.to(torch.bfloat16))


def _bf16_activations():
    """Training-dtype activations entering the MHC layers from BF16 blocks."""
    return torch.tensor(
        [[[[0.2, -0.4, 0.6], [0.8, -1.0, 1.2]], [[-0.3, 0.5, 0.7], [1.1, -0.9, 0.4]]]],
        dtype=torch.bfloat16,
    )


def _sinkhorn_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.reshape(*comb.shape[:-1], hc_mult, hc_mult)
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    # This order is intentional: the production contract adds eps after row
    # normalization, while the later column normalization uses eps in its
    # denominator.  A larger test epsilon makes an accidental reordering fail.
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def test_hc_pre_matches_independent_cpu_reference():
    config = HcPre.Config(hc_mult=2, dim=3, sinkhorn_iters=4, eps=0.05, norm_eps=1e-6)
    module = HcPre(config)
    _set_hc_pre_parameters(module)
    x = torch.tensor(
        [[[[0.2, -0.4, 0.6], [0.8, -1.0, 1.2]], [[-0.3, 0.5, 0.7], [1.1, -0.9, 0.4]]]],
        dtype=torch.float32,
    )

    actual, actual_post, actual_comb = module(x)

    flat = x.flatten(2)
    inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.norm_eps)
    mixes = torch.nn.functional.linear(flat, module.hc_fn) * inverse_rms
    expected_pre, expected_post, expected_comb = _sinkhorn_reference(
        mixes,
        module.hc_scale,
        module.hc_base,
        config.hc_mult,
        config.sinkhorn_iters,
        config.eps,
    )
    assert expected_comb.std() > 0.05
    expected = torch.sum(expected_pre.unsqueeze(-1) * x, dim=2)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_post, expected_post, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_comb, expected_comb, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hc_post_matches_explicit_mixing_equation(dtype):
    module = HcPost(HcPost.Config())
    x = torch.tensor([[[0.2, -0.4, 0.6], [0.8, -1.0, 1.2]]], dtype=dtype)
    residual = torch.tensor(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[-0.3, 0.2, 0.7], [0.9, -0.8, 0.1]]]],
        dtype=dtype,
    )
    post = torch.tensor([[[0.75, 1.25], [1.1, 0.9]]], dtype=torch.float32)
    comb = torch.tensor(
        [[[[0.7, 0.3], [0.2, 0.8]], [[0.6, 0.4], [0.1, 0.9]]]],
        dtype=torch.float32,
    )

    actual = module(x, residual, post, comb)
    expected = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    tolerance = 2e-3 if dtype is torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected.to(dtype), atol=tolerance, rtol=tolerance)


def test_hc_head_matches_independent_cpu_reference():
    config = HcHead.Config(hc_mult=2, dim=3, norm_eps=1e-6, eps=1e-6)
    module = HcHead(config)
    _set_hc_head_parameters(module)
    x = torch.tensor(
        [[[[0.2, -0.4, 0.6], [0.8, -1.0, 1.2]], [[-0.3, 0.5, 0.7], [1.1, -0.9, 0.4]]]],
        dtype=torch.float32,
    )

    actual = module(x)

    flat = x.flatten(2)
    inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.norm_eps)
    mixes = torch.nn.functional.linear(flat, module.hc_fn) * inverse_rms
    pre = torch.sigmoid(mixes * module.hc_scale + module.hc_base) + config.eps
    expected = torch.sum(pre.unsqueeze(-1) * x, dim=2)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_hc_pre_matches_fp64_reference_with_bf16_activations():
    """FP32 parameter storage computes the FP32 math on BF16 activations.

    The per-FQN FSDP policy keeps ``hc_*`` parameters in FP32; the module's
    own ``.float()`` casts promote the BF16 activations. The result must still
    be the exact FP32 equation, not a BF16 approximation of it.
    """
    config = HcPre.Config(hc_mult=2, dim=3, sinkhorn_iters=4, eps=0.05, norm_eps=1e-6)
    module = HcPre(config)
    _set_hc_pre_parameters(module)
    x = _bf16_activations()

    actual, actual_post, actual_comb = module(x)

    flat = x.flatten(2).double()
    inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.norm_eps)
    mixes = torch.nn.functional.linear(flat, module.hc_fn.double()) * inverse_rms
    expected_pre, expected_post, expected_comb = _sinkhorn_reference(
        mixes,
        module.hc_scale.double(),
        module.hc_base.double(),
        config.hc_mult,
        config.sinkhorn_iters,
        config.eps,
    )
    expected = torch.sum(expected_pre.unsqueeze(-1) * x.double(), dim=2)

    assert torch.equal(actual, expected.to(torch.bfloat16))
    torch.testing.assert_close(actual_post, expected_post.float(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_comb, expected_comb.float(), atol=2e-6, rtol=2e-6)


def test_hc_pre_outputs_match_for_fp32_and_bf16_parameter_storage():
    """Only the parameter round trip differs, not the computation.

    The previous FSDP policy exposed BF16 parameter views that ``.float()``
    promoted back to FP32; the per-FQN policy keeps FP32 storage. With equal
    values both must produce bitwise-identical outputs.
    """
    fp32_storage = HcPre(HcPre.Config(hc_mult=2, dim=3, sinkhorn_iters=4, eps=0.05, norm_eps=1e-6))
    _set_hc_pre_parameters(fp32_storage)
    _round_parameters_to_bf16(fp32_storage)
    bf16_storage = copy.deepcopy(fp32_storage)
    _store_parameters_as_bf16(bf16_storage)

    x = _bf16_activations()
    fp32_y, fp32_post, fp32_comb = fp32_storage(x)
    bf16_y, bf16_post, bf16_comb = bf16_storage(x)

    assert torch.equal(fp32_y, bf16_y)
    assert torch.equal(fp32_post, bf16_post)
    assert torch.equal(fp32_comb, bf16_comb)


def test_hc_head_outputs_match_for_fp32_and_bf16_parameter_storage():
    fp32_storage = HcHead(HcHead.Config(hc_mult=2, dim=3, norm_eps=1e-6, eps=1e-6))
    _set_hc_head_parameters(fp32_storage)
    _round_parameters_to_bf16(fp32_storage)
    bf16_storage = copy.deepcopy(fp32_storage)
    _store_parameters_as_bf16(bf16_storage)

    x = _bf16_activations()

    assert torch.equal(fp32_storage(x), bf16_storage(x))
