# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Published DeepSeek mixing semantics through the normal plugin import path."""

import math

import pytest
import torch
from torchtitan.models.deepseek_v4.mhc import HcPre, HcSplitSinkhorn

import torchtitan_npu  # noqa: F401 -- installs package patches before construction


def _deepseek_reference(mixes, scale, base, *, hc_mult, iterations, eps):
    # DeepSeek-V4-Flash inference/kernel.py at 60d8d707: softmax(logits) + eps.
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult:2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])
    logits = (mixes[..., 2 * hc_mult:] * scale[2] + base[2 * hc_mult :]).reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    combination = logits.softmax(dim=-1) + eps
    for iteration in range(iterations):
        if iteration:
            combination = combination / (combination.sum(dim=-1, keepdim=True) + eps)
        combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    return pre, post, combination


def _weighted_sum(outputs):
    return sum(
        (value * torch.cos(torch.arange(value.numel(), dtype=value.dtype).reshape_as(value) * 0.7 + index)).sum()
        for index, value in enumerate(outputs)
    )


@pytest.mark.parametrize(
    ("leading_shape", "hc_mult", "iterations", "eps"),
    [((1,), 1, 1, 1e-5), ((2, 3), 2, 3, 1e-4), ((2,), 4, 20, 1e-6)],
    ids=["singleton", "batched-two-branches", "four-branches-default-epsilon"],
)
def test_split_sinkhorn_matches_deepseek_outputs_and_input_gradients(leading_shape, hc_mult, iterations, eps):
    width = (2 + hc_mult) * hc_mult
    values = torch.arange(math.prod(leading_shape) * width, dtype=torch.float64)
    mixes = (1.7 * torch.sin(values * 0.43)).reshape(*leading_shape, width).requires_grad_()
    scale = torch.tensor([0.6, 1.7, 0.8], dtype=torch.float64, requires_grad=True)
    base = (0.3 * torch.cos(torch.arange(width, dtype=torch.float64) * 0.37)).requires_grad_()
    inputs = (mixes, scale, base)
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    sinkhorn = HcSplitSinkhorn.Config(hc_mult=hc_mult, sinkhorn_iters=iterations, eps=eps).build()

    actual = sinkhorn(*inputs)
    expected = _deepseek_reference(*reference_inputs, hc_mult=hc_mult, iterations=iterations, eps=eps)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-12, atol=1e-12)

    actual_gradients = torch.autograd.grad(_weighted_sum(actual), inputs)
    expected_gradients = torch.autograd.grad(_weighted_sum(expected), reference_inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12)


def test_hc_pre_uses_reference_sinkhorn_for_projected_branches():
    hc_mult, dim, iterations, eps = 2, 3, 2, 2e-3
    pre = HcPre.Config(dim=dim, hc_mult=hc_mult, sinkhorn_iters=iterations, eps=eps, norm_eps=3e-3).build()
    with torch.no_grad():
        pre.hc_fn.copy_(torch.sin(torch.arange(pre.hc_fn.numel()).reshape_as(pre.hc_fn) * 0.31))
        pre.hc_scale.copy_(torch.tensor([0.7, 1.2, 0.9]))
        pre.hc_base.copy_(0.4 * torch.cos(torch.arange(pre.hc_base.numel()) * 0.29))
    x = torch.linspace(-1.2, 1.7, 3 * hc_mult * dim).reshape(3, hc_mult, dim).requires_grad_()
    inputs = (x, pre.hc_fn, pre.hc_scale, pre.hc_base)
    expected_x, expected_weight, expected_scale, expected_base = (
        value.detach().clone().requires_grad_()
        for value in inputs
    )

    actual = pre(x)
    flattened = expected_x.flatten(-2)
    mixes = (flattened @ expected_weight.T) / torch.sqrt(flattened.square().mean(-1, keepdim=True) + 3e-3)
    expected_pre, expected_post, expected_comb = _deepseek_reference(
        mixes, expected_scale, expected_base, hc_mult=hc_mult, iterations=iterations, eps=eps
    )
    expected = ((expected_pre.unsqueeze(-1) * expected_x).sum(-2), expected_post, expected_comb)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-6, atol=2e-7)

    actual_gradients = torch.autograd.grad(_weighted_sum(actual), inputs)
    expected_gradients = torch.autograd.grad(
        _weighted_sum(expected), (expected_x, expected_weight, expected_scale, expected_base)
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=2e-6)
