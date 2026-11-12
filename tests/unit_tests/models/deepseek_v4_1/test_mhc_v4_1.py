# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folded-token mHC outputs and gradients against independent FP64 arithmetic."""

import pytest
import torch

from tests.unit_tests.models.deepseek_v4_1.mhc_reference import hc_post_reference, hc_pre_reference
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre


@pytest.fixture(autouse=True)
def isolated_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def _hc_pre(hc: int, dim: int) -> HcPre:
    module = HcPre.Config(hc_mult=hc, dim=dim, sinkhorn_iters=4, hc_eps=1e-6, norm_eps=1e-6).build()
    with torch.no_grad():
        for name in ("hc_fn", "hc_base", "hc_scale"):
            module.get_parameter(name).normal_(0.0, 0.02)
    return module


def _hc_post() -> HcPost:
    return HcPost.Config().build()


def test_hcpre_forward_contract():
    """The collapsed stream and the coefficients keep their shapes and order."""
    length, hc, dim = 5, 2, 3
    hc_pre = _hc_pre(hc, dim)
    x = torch.randn(length, hc, dim)

    pre_mix = HcPre.identity_pre_mix(x, hc)
    assert pre_mix.shape == (length, hc)
    torch.testing.assert_close(pre_mix[..., 0], torch.ones(length), rtol=0, atol=0)
    torch.testing.assert_close(pre_mix[..., 1:], torch.zeros(length, hc - 1), rtol=0, atol=0)

    y, pre, post, comb = hc_pre(x, pre_mix)
    assert y.shape == (length, dim)
    assert pre.shape == (length, hc)
    assert post.shape == (length, hc)
    assert comb.shape == (length, hc, hc)
    # Sinkhorn keeps comb doubly stochastic over the branches.
    torch.testing.assert_close(comb.sum(-1), torch.ones(length, hc), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comb.sum(-2), torch.ones(length, hc), rtol=1e-4, atol=1e-4)


def test_matches_the_explicit_branch_sum():
    torch.manual_seed(0)
    length, hc, dim = 5, 4, 6
    y = torch.randn(length, dim)
    residual = torch.randn(length, hc, dim)
    post = torch.rand(length, hc)
    comb = torch.randn(length, hc, hc)

    got = _hc_post()(y, residual, post, comb)

    torch.testing.assert_close(got, hc_post_reference(y, residual, post, comb).float(), rtol=1e-5, atol=1e-6)


def test_permuted_comb_permutes_the_branches():
    """A permutation comb must permute the residual branches.

    A permutation has comb row sums of one, so a contraction over the output
    axis returns the residual unchanged instead of swapping the branches.
    """
    length, hc, dim = 3, 2, 4
    residual = torch.randn(length, hc, dim)
    y = torch.zeros(length, dim)
    post = torch.zeros(length, hc)
    comb = torch.zeros(length, hc, hc)
    comb[..., 0, 1] = 1.0  # residual branch 0 feeds output branch 1
    comb[..., 1, 0] = 1.0  # residual branch 1 feeds output branch 0

    got = _hc_post()(y, residual, post, comb)

    torch.testing.assert_close(got, residual.flip(-2), rtol=0, atol=0)


def test_doubly_stochastic_comb_still_mixes():
    """Row sums of one do not make the update the identity."""
    length, hc, dim = 2, 2, 3
    residual = torch.randn(length, hc, dim)
    y = torch.zeros(length, dim)
    post = torch.zeros(length, hc)
    comb = torch.full((length, hc, hc), 0.5)

    got = _hc_post()(y, residual, post, comb)

    expected = 0.5 * (residual[..., 0, :] + residual[..., 1, :])
    expected = expected.unsqueeze(-2).expand(length, hc, dim)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_tokens_are_independent():
    """One token's comb must not change another token's output."""
    torch.manual_seed(4)
    length, hc, dim = 4, 3, 5
    y = torch.randn(length, dim)
    residual = torch.randn(length, hc, dim)
    post = torch.rand(length, hc)
    comb = torch.rand(length, hc, hc)

    base = _hc_post()(y, residual, post, comb)
    perturbed = comb.clone()
    perturbed[0] += 0.5
    changed = _hc_post()(y, residual, post, perturbed)

    assert not torch.allclose(changed[0], base[0]), "the perturbed token had no effect"
    torch.testing.assert_close(changed[1:], base[1:], rtol=0, atol=0)


@pytest.mark.parametrize(("tokens", "branches"), ((1, 2), (5, 4)), ids=("one-token", "four-branches"))
def test_hcpre_outputs_and_gradients_match_per_token_reference(tokens, branches):
    generator = torch.Generator().manual_seed(33)
    module = HcPre.Config(dim=3, hc_mult=branches, sinkhorn_iters=4).build()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
        module.hc_scale.add_(0.7)
    x = torch.randn(tokens, branches, 3, generator=generator, requires_grad=True)
    incoming = torch.rand(tokens, branches, generator=generator, requires_grad=True)
    state = {name: parameter.detach().double().requires_grad_() for name, parameter in module.named_parameters()}
    x_ref, incoming_ref = (value.detach().double().requires_grad_() for value in (x, incoming))

    actual = module(x=x, pre_mix=incoming)
    expected = hc_pre_reference(x_ref, incoming_ref, state, sinkhorn_iters=4, hc_eps=1e-6, norm_eps=1e-6)
    upstream = [torch.randn(value.shape, generator=generator) for value in actual]
    gradients = torch.autograd.grad(actual, (x, incoming, *module.parameters()), upstream)
    reference_gradients = torch.autograd.grad(
        expected, (x_ref, incoming_ref, *state.values()), [value.double() for value in upstream]
    )

    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference.float(), rtol=2e-5, atol=2e-6)
    for value, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(value, reference.float(), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "shapes",
    (
        ((5, 3), (5, 4, 3), (5, 4), (5, 4, 4)),
        ((2, 3, 3), (2, 3, 4, 3), (2, 3, 4), (2, 3, 4, 4)),
        ((3,), (4, 3), (4,), (4, 4)),
        ((2, 1, 3), (2, 3, 4, 3), (1, 3, 4), (4, 4)),
    ),
    ids=("folded", "batched", "single-token", "broadcast-coefficients"),
)
@pytest.mark.parametrize("strided", (False, True), ids=("contiguous", "strided"))
def test_hcpost_input_gradients_match_explicit_branch_sum(shapes, strided):
    generator = torch.Generator().manual_seed(52)
    inputs = []
    for shape in shapes:
        storage_shape = (*shape[:-1], shape[-1] * 2) if strided else shape
        value = torch.randn(storage_shape, generator=generator)
        if strided:
            value = value[..., ::2]
        inputs.append(value.requires_grad_())
    reference_inputs = [value.detach().double().requires_grad_() for value in inputs]
    upstream = torch.randn(shapes[1], generator=generator)

    actual = _hc_post()(y=inputs[0], residual=inputs[1], post=inputs[2], comb=inputs[3])
    expected = hc_post_reference(*reference_inputs)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream.double())

    torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=1e-6)
    for value, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(value, reference.float(), rtol=1e-5, atol=1e-6)
