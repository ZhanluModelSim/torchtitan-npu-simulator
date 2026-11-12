# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-Pass mHC (manifold-constrained Hyper-Connections) for DeepSeek V4.1.

The V4.1 stack carries one stream mix across the whole model: the first
sub-layer collapses with the identity mix (``HcPre.identity_pre_mix``) and
every following sub-layer collapses with the previous one's freshly computed
``pre``; the final hidden state is collapsed once by the last block
(``HcPre.collapse``).  ``HcPre`` and ``HcPost`` follow the reference inference
implementation (``Block.hc_mixes`` / ``Block.hc_post``): the coefficient
derivation, the Sinkhorn balancing and the residual update are the same
operations in the same order.

Shape legend: ``T`` tokens, ``D`` model dimension, ``hc`` residual branches.
The decoder carries ``[T, hc, D]`` streams. Contractions name the trailing
branch axes so independent leading dimensions do not change the arithmetic.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.base.module import Module
from torchtitan.models.deepseek_v4.mhc import HcPost as TorchTitanHcPost


class HcPre(Module):
    """Head-collaboration pre step; owns its mixing parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        sinkhorn_iters: int = 20
        hc_eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_mult = config.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.norm_eps
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    @staticmethod
    def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Return the one-hot stream mix used at the input of the stack.

        Args:
            x: Residual branches of shape ``[T, hc, D]``; its leading dims are
                reused for the mix.
            hc_mult: Number of residual branches ``hc``.

        Returns:
            Mixing coefficients ``[T, hc]`` selecting branch 0.
        """
        pre_mix = torch.zeros(
            (*x.shape[:-2], hc_mult),
            device=x.device,
            dtype=torch.float32,
        )
        pre_mix[..., 0] = 1.0
        return pre_mix

    @staticmethod
    def collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the branches into one sublayer input.

        ``[T, hc, D] x [T, hc] -> [T, D]``
        """
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2)
        return y.to(x.dtype)

    def _split_sinkhorn(self, mixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the mixing logits and balance ``comb`` to be doubly stochastic.

        Args:
            mixes: Mixing logits of shape ``[T, (2 + hc) * hc]``.

        Returns:
            ``(pre, post, comb)`` of shapes ``[T, hc]``, ``[T, hc]`` and
            ``[T, hc, hc]``: a row softmax followed by alternating row and
            column normalizations (Sinkhorn), so ``comb`` is doubly stochastic
            over the branches.
        """
        hc_mult = self.hc_mult
        pre_THc, post, comb_TMcc = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb_TMcc.unflatten(-1, (hc_mult, hc_mult))

        pre_THc = torch.sigmoid(pre_THc * self.hc_scale[0] + self.hc_base[:hc_mult]) + self.hc_eps
        post = 2 * torch.sigmoid(post * self.hc_scale[1] + self.hc_base[hc_mult : 2 * hc_mult])
        comb = comb * self.hc_scale[2] + self.hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

        comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
        comb = comb / comb.sum(dim=-1, keepdim=True) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre_THc, post, comb

    def forward(
        self, x: torch.Tensor, pre_mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse the branches and emit the coefficients the next step consumes.

        Args:
            x: Residual branches of shape ``[T, hc, D]``.
            pre_mix: Input-mixing coefficients ``[T, hc]`` produced by the
                previous sub-layer (``identity_pre_mix`` at the stack input).

        Returns:
            ``(y, pre, post, comb)``: the sublayer input ``[T, D]``, the
            input-mixing coefficients the next sublayer collapses with
            ``[T, hc]``, and the ``post`` ``[T, hc]`` / ``comb``
            ``[T, hc, hc]`` consumed by :class:`HcPost`.
        """
        # One RMS statistic per token over the whole flattened hc * D stream.
        flat_TN = x.flatten(-2).float()
        rsqrt_T1 = torch.rsqrt(flat_TN.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat_TN, self.hc_fn) * rsqrt_T1
        pre_THc, post, comb = self._split_sinkhorn(mixes)
        return self.collapse(x, pre_mix), pre_THc, post, comb


class HcPost(TorchTitanHcPost):
    """Use upstream post mixing with V4.1's arbitrary leading dimensions."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanHcPost.Config):
        pass

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        y: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            y: Sublayer output of shape ``[T, D]``.
            residual: Residual branches entering the sublayer, ``[T, hc, D]``.
            post: Output-mixing coefficients ``[T, hc]``, scaling ``y`` onto each branch.
            comb: Branch-mixing coefficients ``[T, hc, hc]``; entry ``(m, q)`` weights
                residual branch ``m`` into output branch ``q``.

        Returns:
            Updated residual branches of shape ``[T, hc, D]``.

        Higher-rank inputs flatten their leading token dimensions around the upstream
        implementation, which contracts the branch axis at ``dim=1``.
        """
        if y.ndim == 2 and residual.ndim == 3 and post.ndim <= 2 and comb.ndim <= 3:
            return super().forward(y, residual, post, comb)

        # Upstream contracts the branch axis at dim=1 of [T, hc, hc, D].
        # Fold only token dimensions so its reduction sees the same branch order.
        branches, dim = residual.shape[-2:]
        leading_shape = torch.broadcast_shapes(y.shape[:-1], residual.shape[:-2], post.shape[:-1], comb.shape[:-2])
        mixed = super().forward(
            y.expand(*leading_shape, dim).reshape(-1, dim),
            residual.expand(*leading_shape, branches, dim).reshape(-1, branches, dim),
            post.expand(*leading_shape, branches).reshape(-1, branches),
            comb.expand(*leading_shape, branches, branches).reshape(-1, branches, branches),
        )
        return mixed.reshape(*leading_shape, branches, dim)
