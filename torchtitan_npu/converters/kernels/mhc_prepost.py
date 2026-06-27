# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
import torch_npu
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
from torchtitan_npu.ops.triton import MHCPostTriton, MHCPreOnlyTriton, MHCPreTriton
from torchtitan_npu.tools.device import get_npu_device_type

logger = logging.getLogger(__name__)


def _none_grads(count: int) -> tuple[None, ...]:
    return (None,) * count


def _to_local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class MHCPre(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, *args):
        x, weight, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps = args
        weight = weight.to(torch.float32)
        hc_scale = hc_scale.to(torch.float32)
        hc_base = hc_base.to(torch.float32)
        (
            y,
            post,
            comb_frag,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        ) = torch_npu.npu_hc_pre(
            x=x,
            hc_fn=weight,
            hc_scale=hc_scale,
            hc_base=hc_base,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            need_grad=True,
        )

        ctx.save_for_backward(
            x,
            weight,
            hc_scale,
            hc_base,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        )
        ctx.hc_eps = hc_eps
        return y, post, comb_frag

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_y, grad_h_post, grad_h_res = grad_outputs
        (
            x,
            weight,
            hc_scale,
            hc_base,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        ) = ctx.saved_tensors

        grad_x, grad_phi, grad_alpha, grad_bias = torch_npu.npu_mhc_pre_sinkhorn_grad(
            grad_hin=grad_y,
            grad_h_post=grad_h_post,
            grad_h_res=grad_h_res,
            x=x,
            phi=weight,
            alpha=hc_scale,
            bias=hc_base,
            h_pre=h_pre,
            hc_before_norm=hc_before_norm,
            inv_rms=inv_rms,
            sum_out=sum_out,
            norm_out=norm_out,
            hc_eps=ctx.hc_eps,
        )

        return grad_x, grad_phi, grad_alpha, grad_bias, *_none_grads(4)


class MHCPost(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, *args):
        x, residual, h_post, h_res = args
        y = torch_npu.npu_hc_post(
            x=x,
            residual=residual,
            post=h_post,
            comb=h_res,
        )
        ctx.save_for_backward(x, residual, h_post, h_res)
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        x, residual, h_post, h_res = ctx.saved_tensors
        grad_x, grad_residual, grad_h_post, grad_h_res = torch_npu.npu_mhc_post_grad(
            comb_grad=grad_output,
            F_out=x,
            x_l=residual,
            h_post=h_post,
            h_res=h_res,
        )
        return grad_x, grad_residual, grad_h_post, grad_h_res


class NpuHcPre(HcPre):
    def __init__(self, parent: HcPre):
        # Shallow copy of parent's __dict__ is intentional here:
        # - HcPre attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on HcPre.__init__ parameters (hc_mult, hc_sinkhorn_iters, etc.)
        # - Parent instance already has all attributes properly initialized
        # Note: If HcPre had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ):
        r"""HcPre forward using Triton implementation.


        This function executes the "Pre-Mapping" stage of the mHC architecture. It first flattens
        the input from 4D to 3D, then applies RMSNorm normalization, computes manifold-constrained
        connection weights (`h_pre`, `h_post`, `h_res`) via linear projection and the Sinkhorn-Knopp
        algorithm, and finally aggregates the input using `h_pre` to generate the main branch output.


        Args:
            self: Module instance containing hc_mult, hc_sinkhorn_iters, hc_eps attributes
            x (torch.Tensor):
                Input tensor of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
            hc_fn (torch.Tensor):
                Projection weight matrix of shape `[n * n + 2 * n, n * D]`.
                Used to map input to the hyper-connection space.
            hc_scale (torch.Tensor):
                Branch Alpha parameters of shape `[3]`.
            hc_base (torch.Tensor):
                Branch Beta parameters of shape `[2 * n + n * n]`.


        Returns:
            y (torch.Tensor):
                Main branch output of shape `[B, S, D]`.
            h_post (torch.Tensor):
                Post-processing weight matrix of shape `[B, S, n]`.
            h_res (torch.Tensor):
                Residual weight matrix of shape `[B, S, n, n]`.
        """
        x = x.flatten(2)

        y, h_post, h_res = MHCPreTriton.apply(
            x,  # x
            hc_fn,  # weight
            hc_scale,  # branch_alpha
            hc_base,  # branch_beta
            None,  # norm_gamma
            False,  # mhc_use_gamma
            self.hc_mult,  # num_stream
            self.hc_sinkhorn_iters,  # sinkhorn_iters
            self.hc_eps,  # eps
        )
        return y, h_post, h_res


class NpuHcPreFused(HcPre):
    def __init__(self, parent: HcPre):
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ):
        return MHCPre.apply(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )


class NpuHcPost(HcPost):
    def __init__(self, parent: HcPost):
        # Shallow copy of parent's __dict__ is intentional here:
        # - HcPost attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on HcPost.__init__ parameters
        # - Parent instance already has all attributes properly initialized
        # Note: If HcPost had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ):
        r"""NpuHcPost forward using Triton implementation.


        This function executes the "Post-Mapping" stage of the mHC architecture. It flattens the
        residual from 4D to 3D, then utilizes the weights generated in the pre-stage (`h_post` and `h_res`)
        to perform a manifold-constrained weighted fusion of the current input `x` and the `residual`.


        Args:
            self: Module instance
            x (torch.Tensor):
                Current layer main input of shape `[B, S, D]`.
            residual (torch.Tensor):
                Residual input of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
            post (torch.Tensor):
                Post-processing weights of shape `[B, S, n]`.
            comb (torch.Tensor):
                Residual weights of shape `[B, S, n, n]`.


        Returns:
            y (torch.Tensor):
                Fused output tensor of shape `[B, S, N, D]`.
        """
        dim_b, dim_s, dim_n, dim_d = residual.shape
        residual = residual.flatten(2)

        y = MHCPostTriton.apply(
            x,  # x
            residual,  # residual
            post,  # h_post
            comb,  # h_res
        )

        y = y.view(dim_b, dim_s, dim_n, dim_d)
        return y


class NpuHcPostFused(HcPost):
    def __init__(self, parent: HcPost):
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ):
        dim_b, dim_s, dim_n, dim_d = residual.shape
        y = MHCPost.apply(x, residual, post, comb)
        return y.view(dim_b, dim_s, dim_n, dim_d)


class NpuHcHead(HcHead):
    def __init__(self, parent: HcHead):
        # Shallow copy of parent's __dict__ is intentional here:
        # - HcHead attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on HcHead.__init__ parameters (norm_eps, hc_eps)
        # - Parent instance already has all attributes properly initialized
        # Note: If HcHead had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        r"""Lightweight MHC Pre-Aggregation Function (Head forward).


        Similar to `hc_pre`, but this function does not return the intermediate Sinkhorn states
        (`h_post`, `h_res`), returning only the weighted aggregated output. The input is flattened
        from 4D to 3D before processing.


        Args:
            self: Module instance containing hc_mult, hc_eps attributes
            x (torch.Tensor):
                Input tensor of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.


        Returns:
            y (torch.Tensor):
                Weighted aggregated output of shape `[B, S, D]`.
        """
        if isinstance(x, DTensor):
            raise ValueError("NpuHcHead expects local tensor input; apply HcHeadParallelStyle with local TP input.")
        x = x.flatten(2)
        hc_head_fn = _to_local_tensor(self.hc_head_fn)
        hc_head_scale = _to_local_tensor(self.hc_head_scale)
        hc_head_base = _to_local_tensor(self.hc_head_base)

        y = MHCPreOnlyTriton.apply(
            x,  # x
            hc_head_fn,  # weight
            hc_head_scale,  # branch_alpha
            hc_head_base,  # branch_beta
            None,  # norm_gamma
            False,  # mhc_use_gamma
            self.hc_eps,  # eps
        )
        return y


class MHCPreConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        use_fused_kernel = get_npu_device_type() == "A5"
        if use_fused_kernel:
            # pyrefly: ignore [missing-import]
            import custom_ops  # noqa: F401

        hc_pre_cls = NpuHcPreFused if use_fused_kernel else NpuHcPre
        for name, module in list(model.named_modules()):
            if isinstance(module, HcPre):
                replace_module_with_name(model, name, hc_pre_cls(module))
                logger.info("[MHCPreConverter] [HcPre forward] Applied.")


@register_model_converter("npu_mhc_pre")
class MHCPrePostModelConfig(ModelCustomConfig):
    model_converter = MHCPreConverter


class MHCPostConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        use_fused_kernel = get_npu_device_type() == "A5"
        if use_fused_kernel:
            # pyrefly: ignore [missing-import]
            import custom_ops  # noqa: F401

        hc_post_cls = NpuHcPostFused if use_fused_kernel else NpuHcPost
        for name, module in list(model.named_modules()):
            if isinstance(module, HcPost):
                replace_module_with_name(model, name, hc_post_cls(module))
                logger.info("[MHCPostConverter] [HcPost forward] Applied.")

            if not use_fused_kernel and isinstance(module, HcHead):
                replace_module_with_name(model, name, NpuHcHead(module))
                logger.info("[MHCPostConverter] [HcHead forward] Applied.")


@register_model_converter("npu_mhc_post")
class MHCPostModelConfig(ModelCustomConfig):
    model_converter = MHCPostConverter
