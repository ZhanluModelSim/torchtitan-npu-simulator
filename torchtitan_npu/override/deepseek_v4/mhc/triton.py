# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 MHC pre/post/head with hand-written Triton kernels."""

from dataclasses import dataclass

import torch
import torch_npu

from torchtitan_npu.models.deepseek_v4.mhc import HcHead, HcPost, HcPre
from torchtitan_npu.ops.triton.mhc import (
    mhc_post_bmm1_op,
    mhc_post_bmm2_op,
    mhc_pre_bmm_op,
    mhc_pre_only_sinkhorn_op,
    mhc_pre_sinkhorn_op,
)


class TritonHcPre(HcPre):
    """HcPre backed by the hand-written Triton sinkhorn/bmm kernels."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Kernels consume the flattened 3-D form ``[B, S, hc_mult * dim]``.
        x = x.flatten(2)
        B, S, nD = x.shape
        dtype = x.dtype

        x = x.float()
        weight = self.hc_fn.float().t()
        branch_alpha = self.hc_scale.float()
        branch_beta = self.hc_base.float()

        # RMSNorm and linear projection stay in the composition layer; the
        # sinkhorn and BMM steps are the registered Triton custom ops.
        x_flat = x.reshape(-1, nD)
        norm_gamma = torch.ones(nD, device=x.device, dtype=torch.float32)
        x_norm_flat, _rstd = torch_npu.npu_rms_norm(x_flat, gamma=norm_gamma, epsilon=self.eps)
        x_norm_mat = x_norm_flat.reshape(B, S, nD)
        x_proj = torch.matmul(x_norm_mat, weight)

        h_pre, post, comb = mhc_pre_sinkhorn_op(
            x_proj,
            branch_alpha,
            branch_beta,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.eps,
        )

        x_unflatten = x.unflatten(dim=-1, sizes=(self.hc_mult, -1))
        y = mhc_pre_bmm_op(h_pre, x_unflatten)
        return y.to(dtype), post, comb


class TritonHcPost(HcPost):
    """HcPost backed by the hand-written Triton bmm kernels."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        input_x = x
        residual_shape = residual.shape
        post = post.float()
        comb = comb.permute(0, 1, 3, 2).float()
        B, S, D = x.shape
        dtype = x.dtype
        N = post.shape[-1]
        x = x.float()
        residual = residual.flatten(2).float()

        # Keep the shape checks next to the two post-BMM custom ops.
        if residual.shape[:-1] != (B, S):
            raise ValueError("residual shape mismatch")
        if residual.shape[-1] != N * D:
            raise ValueError(f"residual last dim {residual.shape[-1]} != N*D={N * D}")
        if comb.shape != (B, S, N, N):
            raise ValueError(f"h_res shape {comb.shape} != ({B},{S},{N},{N})")

        bmm1 = mhc_post_bmm1_op(x, post)
        residual_unflat = residual.view(B, S, N, D)
        bmm2 = mhc_post_bmm2_op(comb, residual_unflat)
        result_flat = bmm1 + bmm2
        y = result_flat.view(B, S, N * D).to(dtype)
        return y.view(residual_shape).type_as(input_x)


class TritonHcHead(HcHead):
    """HcHead backed by ``mhc_pre_only``; no CANN fused counterpart exists."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcHead.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # ``HcHead`` only uses hc_mult to size its parameters and drops it;
        # the Triton entry needs it as ``num_stream``.
        self.hc_mult = config.hc_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2)
        B, S, nD = x.shape
        dtype = x.dtype

        x = x.float()
        weight = self.hc_fn.float().t()
        branch_alpha = self.hc_scale.float()
        branch_beta = self.hc_base.float()

        x_flat = x.reshape(-1, nD)
        norm_gamma = torch.ones(nD, device=x.device, dtype=torch.float32)
        x_norm_flat, _rstd = torch_npu.npu_rms_norm(x_flat, gamma=norm_gamma, epsilon=self.eps)
        x_norm_mat = x_norm_flat.reshape(B, S, nD)
        x_proj = torch.matmul(x_norm_mat, weight)

        h_pre = mhc_pre_only_sinkhorn_op(
            x_proj,
            branch_alpha,
            branch_beta,
            hc_mult=self.hc_mult,
            eps=self.eps,
        )

        x_unflatten = x.unflatten(dim=-1, sizes=(self.hc_mult, -1))
        y = mhc_pre_bmm_op(h_pre, x_unflatten)
        return y.to(dtype)
