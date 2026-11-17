# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent mHC stages; preserve V4.1's cross-layer pre-mix graph."""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

import torchtitan_npu.ops.ascendc.mhc  # noqa: F401
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre


class AscV41HcPre(HcPre):
    """Fuse only Sinkhorn; inherited collapse still uses the caller's pre mix."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def _split_sinkhorn(self, mixes_BLM):
        n = self.hc_mult
        pre, post, comb = mixes_BLM.split([n, n, n * n], dim=-1)
        pre = torch.sigmoid(pre * self.hc_scale[0] + self.hc_base[:n]) + self.hc_eps
        post = 2 * torch.sigmoid(post * self.hc_scale[1] + self.hc_base[n : 2 * n])
        comb = (comb * self.hc_scale[2] + self.hc_base[2 * n :]).unflatten(-1, (n, n)).contiguous()
        comb, _, _ = torch_npu.npu_mhc_sinkhorn(comb, eps=self.hc_eps, num_iters=self.sinkhorn_iters, out_flag=1)
        return pre, post, comb


class AscV41HcPost(HcPost):
    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(self, y_BLD, residual_BLHcD, post_BLHc, comb_BLHcHc):
        # Sequence-parallel linears hand the collapsed stream back batch-folded
        # (``[T, D]`` with B=1); the reference arithmetic broadcasts that
        # silently, but the kernel requires the explicit ``[B, T, D]`` form.
        if y_BLD.ndim == 2:
            y_BLD = y_BLD.unsqueeze(0)
        return torch.ops.cann_ops_transformer.mhc_post(residual_BLHcD, comb_BLHcHc, y_BLD, post_BLHc)


@override(target=HcPre.Config, exact=True, description="V4.1 single-pass mHC with native Sinkhorn")
def asc_sinkhorn(cfg: HcPre.Config) -> AscV41HcPre.Config:
    return derive(cfg, AscV41HcPre.Config)


@override(target=HcPost.Config, exact=True, description="V4.1 AscendC mHC post")
def asc_hc_post(cfg: HcPost.Config) -> AscV41HcPost.Config:
    return derive(cfg, AscV41HcPost.Config)
