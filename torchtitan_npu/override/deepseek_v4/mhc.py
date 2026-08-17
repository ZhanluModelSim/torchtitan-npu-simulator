# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 MHC pre/post with NPU operators."""

from dataclasses import dataclass

import torch
import torch_npu
from cann_ops_transformer import ops as cann_ops
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.mhc import HcPost, HcPre


class CANNHcPre(HcPre):
    """HcPre backed by split ``torch_npu`` operators.

    The modern model-dir ``HcPre`` owns its mixing parameters and calls
    ``forward(x)``. Training uses ``npu_mhc_pre`` followed by
    ``npu_mhc_sinkhorn`` so autograd takes the split backward path instead of
    the slower fused ``mhc_pre_sinkhorn`` backward kernel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        h_in, h_post, h_res, _, _, _ = torch_npu.npu_mhc_pre(
            x,
            self.hc_fn.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            norm_eps=self.norm_eps,
            hc_eps=self.eps,
            out_flag=1,
        )
        h_res, _, _ = torch_npu.npu_mhc_sinkhorn(
            h_res,
            eps=self.eps,
            num_iters=self.sinkhorn_iters,
            out_flag=1,
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
    description="NPU DeepSeek-V4 HcPre via split torch_npu operators",
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
