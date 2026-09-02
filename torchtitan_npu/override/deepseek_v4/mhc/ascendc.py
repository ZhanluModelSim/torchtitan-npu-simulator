# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 MHC pre/post with NPU operators."""

from dataclasses import dataclass

import torch
import torch_npu

import torchtitan_npu.ops.ascendc.mhc  # noqa: F401
from torchtitan_npu.models.deepseek_v4.mhc import HcPost, HcPre
from torchtitan_npu.override import _IS_A5


class AscHcPre(HcPre):
    """HcPre backed by the hardware-specific AscendC implementation.

    The modern model-dir ``HcPre`` owns its mixing parameters and calls
    ``forward(x)``. Ascend A3 uses the fused
    ``cann_ops_transformer.mhc_pre_sinkhorn`` path; Ascend A5 uses
    ``torch_npu.npu_mhc_pre`` followed by ``torch_npu.npu_mhc_sinkhorn``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._use_split_backend = _IS_A5

    def _forward_fused(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        h_in, h_post, h_res, *_ = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn(
            x,
            self.hc_fn.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.eps,
            self.norm_eps,
            True,
        )
        return h_in.to(dtype), h_post, h_res

    def _forward_split(
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
            inner_precise=1,
        )
        h_res, _, _ = torch_npu.npu_mhc_sinkhorn(
            h_res,
            eps=self.eps,
            num_iters=self.sinkhorn_iters,
            out_flag=1,
        )
        return h_in.to(dtype), h_post, h_res

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Once A5's fused mhc_pre_sinkhorn is performance-optimized, route A3/A5 through it uniformly.
        h_in, h_post, h_res = self._forward_split(x) if self._use_split_backend else self._forward_fused(x)
        h_res = h_res.reshape(*x.shape[:2], self.hc_mult, self.hc_mult)
        return h_in, h_post, h_res


class AscHcPost(HcPost):
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
        return torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post)
