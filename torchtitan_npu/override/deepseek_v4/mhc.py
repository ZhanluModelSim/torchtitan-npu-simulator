# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 MHC pre/post with fused NPU operators."""

from dataclasses import dataclass

import torch
from cann_ops_transformer import ops as cann_ops
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.mhc import HcPost, HcPre


class CANNHcPre(HcPre):
    """HcPre backed by ``cann_ops_transformer.ops.mhc_pre_sinkhorn``.

    The modern model-dir ``HcPre`` owns its mixing parameters and calls
    ``forward(x)``; the fused op consumes them internally (linear + RMS
    scaling + sigmoid + sinkhorn), mirroring the eager implementation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The block feeds the 4-D ``[B, S, hc_mult, dim]`` form (the eager
        # path flattens to ``[B, S, hc_mult * dim]`` for its linear); the
        # fused op consumes the 4-D form directly (x: FLOAT16/BFLOAT16).
        dtype = x.dtype
        h_in, h_post, h_res = cann_ops.mhc_pre_sinkhorn(
            x,
            self.hc_fn.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.eps,
            self.norm_eps,
        )
        h_res = h_res.reshape(*x.shape[:2], self.hc_mult, self.hc_mult)
        return h_in.to(dtype), h_post, h_res


class CANNHcPost(HcPost):
    """HcPost backed by ``cann_ops_transformer.ops.mhc_post``."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return cann_ops.mhc_post(residual, comb, x, post)


@override(
    target=HcPre.Config,
    exact=True,
    description=("NPU fused DeepSeek-V4 HcPre via cann_ops_transformer.ops.mhc_pre_sinkhorn"),
)
def cann_hc_pre(cfg: HcPre.Config) -> CANNHcPre.Config:
    return derive(cfg, CANNHcPre.Config)


@override(
    target=HcPost.Config,
    exact=True,
    description=("NPU fused DeepSeek-V4 HcPost via cann_ops_transformer.ops.mhc_post"),
)
def cann_hc_post(cfg: HcPost.Config) -> CANNHcPost.Config:
    return derive(cfg, CANNHcPost.Config)
