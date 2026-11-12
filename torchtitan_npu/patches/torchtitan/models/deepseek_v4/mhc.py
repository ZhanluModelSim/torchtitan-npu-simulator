# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: pending; tracked by https://github.com/pytorch/torchtitan/issues/5975
# Owner: torchtitan_npu.patches.torchtitan.models.deepseek_v4.mhc

"""Preserve DeepSeek's additive epsilon in the first Sinkhorn normalization.

Remove this patch when the pinned TorchTitan version includes the reference
normalization tracked in issue #5975.
"""

import torch
import torchtitan.models.deepseek_v4.mhc


def _hc_split_sinkhorn_forward(
    self: torchtitan.models.deepseek_v4.mhc.HcSplitSinkhorn,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = self.hc_mult
    mid = 2 * hc_mult
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].view(*([1] * (pre.ndim - 1)), hc_mult)) + self.eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult:mid].view(*([1] * (post.ndim - 1)), hc_mult))
    comb = comb * hc_scale[2] + hc_base[mid:].view(*([1] * (comb.ndim - 2)), hc_mult, hc_mult)

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    # The reference adds a positive entry floor before balancing; see #5975.
    comb = comb / comb.sum(dim=-1, keepdim=True) + self.eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
    for _ in range(self.sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
    return pre, post, comb


def _hc_post_forward(self, x, residual, post, comb):
    """Sum the combination over the HC branch axis, not a fixed ``dim=1``.

    Upstream's unbatched ``[T, hc, hc, D]`` sees that axis at ``dim=1``; a leading
    batch axis moves it, so address it from the right.
    """
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
    return y.type_as(x)


def apply() -> None:
    """Update the existing upstream class, retaining its config and callers."""
    torchtitan.models.deepseek_v4.mhc.HcSplitSinkhorn.forward = _hc_split_sinkhorn_forward
    torchtitan.models.deepseek_v4.mhc.HcPost.forward = _hc_post_forward


apply()
