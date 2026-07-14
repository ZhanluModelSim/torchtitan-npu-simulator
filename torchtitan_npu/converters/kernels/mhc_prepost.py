# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import logging
from typing import Any

import torch
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

_mhc_ops_module: Any | None = None


def _mhc_ops() -> Any:
    global _mhc_ops_module
    if _mhc_ops_module is None:
        try:
            module: Any = importlib.import_module("cann_ops_transformer")
        except ImportError as exc:
            raise RuntimeError("DeepSeekV4 A5 MHC fusion requires the cann_ops_transformer package.") from exc
        _mhc_ops_module = module.ops
    return _mhc_ops_module


def _to_local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _mhc_pre_sinkhorn(
    x: Tensor,
    weight: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    input_x = x
    use_fake_batch = x.dim() == 3
    if use_fake_batch:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise RuntimeError(f"MHC fused kernel expects residual to be TND or BSND, got shape {tuple(x.shape)}.")

    outputs = _mhc_ops().mhc_pre_sinkhorn(
        x.contiguous(),
        weight.to(torch.float32),
        hc_scale.to(torch.float32),
        hc_base.to(torch.float32),
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        norm_eps,
    )
    dim_b, dim_s, dim_n, _ = x.shape
    h_in, h_post, h_res = outputs[:3]
    h_in = h_in.view(dim_b, dim_s, h_in.shape[-1])
    h_post = h_post.view(dim_b, dim_s, h_post.shape[-1])
    h_res = h_res.view(dim_b, dim_s, dim_n, dim_n)
    if use_fake_batch:
        h_in = h_in.squeeze(0)
        h_post = h_post.squeeze(0)
    return h_in.type_as(input_x), h_post, h_res


def _mhc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    use_fake_batch = residual.dim() == 3
    if use_fake_batch:
        residual_bsnd = residual.unsqueeze(0)
        x_bsd = x.unsqueeze(0)
        post_bsn = post.unsqueeze(0)
        comb_bsnn = comb if comb.dim() == 4 else comb.unsqueeze(0)
    elif residual.dim() == 4:
        residual_bsnd = residual
        x_bsd = x
        post_bsn = post
        comb_bsnn = comb
    else:
        raise RuntimeError(f"MHC fused kernel expects residual to be TND or BSND, got shape {tuple(residual.shape)}.")

    output = _mhc_ops().mhc_post(
        residual_bsnd.contiguous(),
        comb_bsnn.contiguous(),
        x_bsd.contiguous(),
        post_bsn.contiguous(),
    )
    if output.dim() == 3:
        output = output.view_as(residual_bsnd)
    if use_fake_batch:
        output = output.squeeze(0)
    return output.type_as(residual)


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
        return _mhc_pre_sinkhorn(
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
        return _mhc_post(x, residual, post, comb).view_as(residual)


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
            _mhc_ops()

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
            _mhc_ops()

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
