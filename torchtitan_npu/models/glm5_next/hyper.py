# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Manifold-Constrained Hyper-Connections (mHC) for GLM-5.3-Flash.

Mirrors ``Glm5NextTextHyperConnection`` / ``Glm5NextTextUnweightedRMSNorm``
in MindSpeed-MM ``modeling_glm5_next.py``. The production NPU path calls
``hc_split_sinkhorn`` (triton ``_hc_split_sinkhorn_kernel_part1/2``); the
0.3.0 capture is ATen level, so this module records the eager math
(sigmoid gates, softmax + Sinkhorn normalization). hc_mult=4 is a hard
production constraint (the triton kernel asserts it).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from torchtitan.protocols.module import Module


def _hc_fn_init(param: torch.Tensor) -> None:
    nn.init.normal_(param, mean=0.0, std=0.02)


class HyperConnection(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        hc_mult: int = 4
        sinkhorn_iters: int = 20
        eps: float = 1e-6
        norm_eps: float = 1e-6

        def __post_init__(self) -> None:
            # Production init (Glm5NextTextHyperConnection): fn ~ N(0, 0.02),
            # base zeros, scale ones.
            self.param_init = {
                "fn": _hc_fn_init,
                "base": torch.nn.init.zeros_,
                "scale": torch.nn.init.ones_,
            }

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.hc_eps = config.eps
        self.norm_eps = config.norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.dim))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))
        # Unit scale for the fused ``npu_rms_norm`` (Glm5NextTextUnweightedRMSNorm
        # runs the norm through the fused kernel with an all-ones weight).
        self.register_buffer("_ones", torch.ones(self.hc_mult * config.dim, dtype=torch.float32), persistent=False)

    def _unweighted_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self._ones.to(x.dtype), epsilon=self.norm_eps)[0]

    def _sinkhorn(self, mixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self._unweighted_rms_norm(hidden_streams.flatten(start_dim=2).float())
        mixes = F.linear(flat, self.fn.float())
        pre, post, comb = self._sinkhorn(mixes)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
