# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek V4.1 mHC modules.

``HcPost`` contracts the residual-branch axis: entry ``(m, q)`` of ``comb``
weights residual branch ``m`` into output branch ``q``::

    out[b, l, q, d] = post[b, l, q] * y[b, l, d]
                      + sum_m comb[b, l, m, q] * residual[b, l, m, d]

The pinned torchtitan keeps a batch dim on the stream, so ``comb * residual``
has shape ``[B, L, hc, hc, D]`` with axes ``(b, l, m, q, d)`` and the
contraction is over ``m`` (``dim=2``).  Contracting ``q`` instead collapses the
update to ``residual * comb.sum(-1)``: a per-branch rescaling that is
approximately the identity, because Sinkhorn drives the comb row sums to one.

``HcPre.forward`` returns ``(y, pre, post, comb)``, the order ``HcPost`` and the
next sub-layer consume.
"""

import torch

from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre


def _hc_pre(hc: int, dim: int) -> HcPre:
    module = HcPre.Config(hc_mult=hc, dim=dim, sinkhorn_iters=4, hc_eps=1e-6, norm_eps=1e-6).build()
    with torch.no_grad():
        for name in ("hc_fn", "hc_base", "hc_scale"):
            module.get_parameter(name).normal_(0.0, 0.02)
    return module


def _hc_post() -> HcPost:
    return HcPost.Config().build()


def _oracle(y, residual, post, comb):
    """Explicit per-branch sum in float64, one term at a time."""
    y64 = y.double()
    residual64 = residual.double()
    post64 = post.double()
    comb64 = comb.double()
    hc = residual.size(-2)
    out = post64.unsqueeze(-1) * y64.unsqueeze(-2)
    for m in range(hc):
        for q in range(hc):
            out[..., q, :] = out[..., q, :] + comb64[..., m, q].unsqueeze(-1) * residual64[..., m, :]
    return out


def test_hcpre_forward_contract():
    """The collapsed stream and the coefficients keep their shapes and order."""
    batch, length, hc, dim = 2, 5, 2, 3
    hc_pre = _hc_pre(hc, dim)
    x = torch.randn(batch, length, hc, dim)

    pre_mix = HcPre.identity_pre_mix(x, hc)
    assert pre_mix.shape == (batch, length, hc)
    torch.testing.assert_close(pre_mix[..., 0], torch.ones(batch, length), rtol=0, atol=0)
    torch.testing.assert_close(pre_mix[..., 1:], torch.zeros(batch, length, hc - 1), rtol=0, atol=0)

    y, pre, post, comb = hc_pre(x, pre_mix)
    assert y.shape == (batch, length, dim)
    assert pre.shape == (batch, length, hc)
    assert post.shape == (batch, length, hc)
    assert comb.shape == (batch, length, hc, hc)
    # Sinkhorn keeps comb doubly stochastic over the branches.
    torch.testing.assert_close(comb.sum(-1), torch.ones(batch, length, hc), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comb.sum(-2), torch.ones(batch, length, hc), rtol=1e-4, atol=1e-4)


def test_matches_the_explicit_branch_sum():
    torch.manual_seed(0)
    batch, length, hc, dim = 2, 5, 4, 6
    y = torch.randn(batch, length, dim)
    residual = torch.randn(batch, length, hc, dim)
    post = torch.rand(batch, length, hc)
    comb = torch.randn(batch, length, hc, hc)

    got = _hc_post()(y, residual, post, comb)

    torch.testing.assert_close(got, _oracle(y, residual, post, comb).float(), rtol=1e-5, atol=1e-6)


def test_permuted_comb_permutes_the_branches():
    """A permutation comb must permute the residual branches.

    A permutation has comb row sums of one, so a contraction over the output
    axis returns the residual unchanged instead of swapping the branches.
    """
    batch, length, hc, dim = 2, 3, 2, 4
    residual = torch.randn(batch, length, hc, dim)
    y = torch.zeros(batch, length, dim)
    post = torch.zeros(batch, length, hc)
    comb = torch.zeros(batch, length, hc, hc)
    comb[..., 0, 1] = 1.0  # residual branch 0 feeds output branch 1
    comb[..., 1, 0] = 1.0  # residual branch 1 feeds output branch 0

    got = _hc_post()(y, residual, post, comb)

    torch.testing.assert_close(got, residual.flip(-2), rtol=0, atol=0)


def test_doubly_stochastic_comb_still_mixes():
    """Row sums of one do not make the update the identity."""
    batch, length, hc, dim = 2, 2, 2, 3
    residual = torch.randn(batch, length, hc, dim)
    y = torch.zeros(batch, length, dim)
    post = torch.zeros(batch, length, hc)
    comb = torch.full((batch, length, hc, hc), 0.5)

    got = _hc_post()(y, residual, post, comb)

    expected = 0.5 * (residual[..., 0, :] + residual[..., 1, :])
    expected = expected.unsqueeze(-2).expand(batch, length, hc, dim)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_tokens_and_batches_are_independent():
    """One token's comb must not change another token's or batch's output."""
    torch.manual_seed(4)
    batch, length, hc, dim = 2, 4, 3, 5
    y = torch.randn(batch, length, dim)
    residual = torch.randn(batch, length, hc, dim)
    post = torch.rand(batch, length, hc)
    comb = torch.rand(batch, length, hc, hc)

    base = _hc_post()(y, residual, post, comb)
    perturbed = comb.clone()
    perturbed[0, 0] += 0.5
    changed = _hc_post()(y, residual, post, perturbed)

    assert not torch.allclose(changed[0, 0], base[0, 0]), "the perturbed token had no effect"
    torch.testing.assert_close(changed[0, 1:], base[0, 1:], rtol=0, atol=0)
    torch.testing.assert_close(changed[1], base[1], rtol=0, atol=0)
