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


class FusedHcPre(HcPre):
    """HcPre backed by ``cann_ops_transformer.ops.mhc_pre_sinkhorn``."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = cann_ops.mhc_pre_sinkhorn(
            x,
            hc_fn.to(torch.float32),
            hc_scale.to(torch.float32),
            hc_base.to(torch.float32),
            self.hc_mult,
            self.sinkhorn.sinkhorn_iters,
            self.sinkhorn.eps,
            self.norm_eps,
        )

        h_in, h_post, h_res = outputs[:3]
        h_res = h_res.reshape(*x.shape[:2], self.hc_mult, self.hc_mult)
        return h_in, h_post, h_res


class FusedHcPost(HcPost):
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
    description=(
        "NPU fused DeepSeek-V4 HcPre via cann_ops_transformer.ops.mhc_pre_sinkhorn"
    ),
)
def npu_mhc_pre(cfg: HcPre.Config) -> FusedHcPre.Config:
    return derive(cfg, FusedHcPre.Config)


@override(
    target=HcPost.Config,
    exact=True,
    description=("NPU fused DeepSeek-V4 HcPost via cann_ops_transformer.ops.mhc_post"),
)
def npu_mhc_post(cfg: HcPost.Config) -> FusedHcPost.Config:
    return derive(cfg, FusedHcPost.Config)
