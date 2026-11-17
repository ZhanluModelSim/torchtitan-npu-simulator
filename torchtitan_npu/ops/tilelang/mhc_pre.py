# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom-op boundary for the TileKernels mHC pre path."""

from typing import Any

import torch
import torch.nn.functional as F
import torch_npu


def _split_backward(
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    input_mixes: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    hc_scale: torch.Tensor,
    pre_eps: float,
    post_mult: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the TileKernels split-mixes formula in PyTorch."""

    hc_mult = pre.shape[-2]
    pre_sigmoid = pre.squeeze(-1) - pre_eps
    post_sigmoid = post.squeeze(-1) / post_mult
    grad_pre_affine = grad_pre.squeeze(-1) * pre_sigmoid * (1.0 - pre_sigmoid)
    grad_post_affine = grad_post.squeeze(-1) * post_mult * post_sigmoid * (1.0 - post_sigmoid)
    grad_comb_affine = grad_comb.flatten(-2)

    grad_input = torch.cat(
        (
            grad_pre_affine * hc_scale[0],
            grad_post_affine * hc_scale[1],
            grad_comb_affine * hc_scale[2],
        ),
        dim=-1,
    )
    reduce_dims = tuple(range(input_mixes.ndim - 1))
    split_index = 2 * hc_mult
    grad_scale = torch.stack(
        (
            (grad_pre_affine * input_mixes[..., :hc_mult]).sum(),
            (grad_post_affine * input_mixes[..., hc_mult:split_index]).sum(),
            (grad_comb_affine * input_mixes[..., split_index:]).sum(),
        )
    )
    grad_base = torch.cat(
        (
            grad_pre_affine.sum(dim=reduce_dims),
            grad_post_affine.sum(dim=reduce_dims),
            grad_comb_affine.sum(dim=reduce_dims),
        )
    )
    return grad_input, grad_scale, grad_base


def _load_split():
    try:
        from tile_kernels.modeling.mhc.ops.pre_split_mixes import mhc_pre_split_mixes
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TileLang HcPre requires the installed tile_kernels package and a loaded CANN environment"
        ) from error
    return mhc_pre_split_mixes


def _load_apply():
    try:
        from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TileLang HcPre requires the installed tile_kernels package and a loaded CANN environment"
        ) from error
    return mhc_pre_apply_mix


@torch.library.custom_op(
    "torchtitan_npu::tilelang_mhc_pre_fwd",
    mutates_args=(),
    device_types="npu",
)
def tilelang_mhc_pre_fwd_op(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run TileKernels split/apply plus the native Sinkhorn runtime path."""

    if x.ndim != 4:
        raise ValueError(f"TileLang HcPre expects [B,S,N,D], got {tuple(x.shape)}")
    x_float = x.flatten(2).float().contiguous()
    rsqrt = torch.rsqrt(x_float.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_float, hc_fn.float()) * rsqrt

    split = _load_split()
    pre, post, comb = split(mixes, hc_scale.float(), hc_base.float(), hc_mult, 2.0, eps)
    comb, sinkhorn_norm, sinkhorn_sum = torch_npu.npu_mhc_sinkhorn(
        comb,
        eps=eps,
        num_iters=sinkhorn_iters,
        out_flag=1,
    )
    layer_input = _load_apply()(x, pre)
    return (
        layer_input.to(x.dtype),
        post.squeeze(-1).clone(),
        comb,
        x_float,
        rsqrt,
        mixes,
        pre,
        post,
        sinkhorn_norm,
        sinkhorn_sum,
    )


@tilelang_mhc_pre_fwd_op.register_fake
def _tilelang_mhc_pre_fwd_fake(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, ...]:
    B, S, N, D = x.shape
    total_mix = 2 * hc_mult + hc_mult * hc_mult
    token_count = B * S
    sinkhorn_norm = torch.empty(
        (token_count * sinkhorn_iters * 4 * hc_mult * hc_mult,),
        dtype=torch.float32,
        device=x.device,
    )
    sinkhorn_sum = torch.empty(
        (token_count * sinkhorn_iters * hc_mult * hc_mult,),
        dtype=torch.float32,
        device=x.device,
    )
    return (
        torch.empty((B, S, D), dtype=x.dtype, device=x.device),
        torch.empty((B, S, N), dtype=torch.float32, device=x.device),
        torch.empty((B, S, N, N), dtype=torch.float32, device=x.device),
        torch.empty((B, S, N * D), dtype=torch.float32, device=x.device),
        torch.empty((B, S, 1), dtype=torch.float32, device=x.device),
        torch.empty((B, S, total_mix), dtype=torch.float32, device=x.device),
        torch.empty((B, S, N, 1), dtype=torch.float32, device=x.device),
        torch.empty((B, S, N, 1), dtype=torch.float32, device=x.device),
        sinkhorn_norm,
        sinkhorn_sum,
    )


@torch.library.custom_op(
    "torchtitan_npu::tilelang_mhc_pre_bwd",
    mutates_args=(),
    device_types="npu",
)
def tilelang_mhc_pre_bwd_op(
    grad_y: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    x_float: torch.Tensor,
    rsqrt: torch.Tensor,
    mixes: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    sinkhorn_norm: torch.Tensor,
    sinkhorn_sum: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile-safe backward for the TileKernels HcPre composition."""

    B, S, nD = x_float.shape
    hc_mult = pre.shape[-2]
    x_unflatten = x_float.view(B, S, hc_mult, nD // hc_mult)
    grad_y_f = grad_y.float()
    grad_pre = (grad_y_f.unsqueeze(-2) * x_unflatten).sum(-1, keepdim=True)
    grad_x_direct = grad_y_f.unsqueeze(-2) * pre

    if grad_comb is None:
        grad_comb = torch.zeros(
            (*pre.shape[:-2], hc_mult, hc_mult),
            dtype=pre.dtype,
            device=pre.device,
        )
    grad_comb_before_sinkhorn = torch.ops.npu.npu_mhc_sinkhorn_backward(
        grad_comb,
        sinkhorn_norm,
        sinkhorn_sum,
    )
    grad_mixes, grad_scale, grad_base = _split_backward(
        grad_pre,
        grad_post,
        grad_comb_before_sinkhorn,
        mixes,
        pre,
        post,
        hc_scale,
        eps,
        2.0,
    )

    x_norm = x_float * rsqrt
    grad_x_norm = torch.matmul(grad_mixes, hc_fn)
    grad_hc_fn = torch.matmul(grad_mixes.reshape(-1, grad_mixes.shape[-1]).t(), x_norm.reshape(-1, nD))
    grad_x_rms = grad_x_norm * rsqrt - x_float * rsqrt.square() * rsqrt * (
        (x_float * grad_x_norm).sum(-1, keepdim=True) / nD
    )
    grad_x = (grad_x_direct.reshape(B, S, nD) + grad_x_rms).to(dtype=x_float.dtype)
    grad_x = grad_x.view(B, S, hc_mult, nD // hc_mult)
    return grad_x, grad_hc_fn, grad_scale, grad_base


@tilelang_mhc_pre_bwd_op.register_fake
def _tilelang_mhc_pre_bwd_fake(
    grad_y: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    x_float: torch.Tensor,
    rsqrt: torch.Tensor,
    mixes: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    sinkhorn_norm: torch.Tensor,
    sinkhorn_sum: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(
            (*x_float.shape[:2], pre.shape[-2], x_float.shape[-1] // pre.shape[-2]),
            dtype=x_float.dtype,
            device=x_float.device,
        ),
        torch.empty_like(hc_fn, dtype=torch.float32),
        torch.empty_like(hc_scale, dtype=torch.float32),
        torch.empty_like(hc_base, dtype=torch.float32),
    )


def _tilelang_mhc_pre_setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
    x, hc_fn, hc_scale, hc_base, _hc_mult, _sinkhorn_iters, eps, _norm_eps = inputs
    _, _, _, x_float, rsqrt, mixes, pre, post, sinkhorn_norm, sinkhorn_sum = output
    ctx.save_for_backward(x_float, rsqrt, mixes, pre, post, sinkhorn_norm, sinkhorn_sum, hc_fn, hc_scale, hc_base)
    ctx.input_dtype = x.dtype
    ctx.eps = eps


def _tilelang_mhc_pre_backward(
    ctx: Any,
    grad_y: torch.Tensor | None,
    grad_post: torch.Tensor | None,
    grad_comb: torch.Tensor | None,
    *_saved_output_grads: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    (
        x_float,
        rsqrt,
        mixes,
        pre,
        post,
        sinkhorn_norm,
        sinkhorn_sum,
        hc_fn,
        hc_scale,
        hc_base,
    ) = ctx.saved_tensors
    hc_mult = pre.shape[-2]
    if grad_y is None:
        grad_y = torch.zeros(
            (*x_float.shape[:2], x_float.shape[-1] // hc_mult),
            dtype=x_float.dtype,
            device=x_float.device,
        )
    if grad_post is None:
        grad_post = torch.zeros_like(post.squeeze(-1))
    if grad_comb is None:
        grad_comb = torch.zeros(
            (*pre.shape[:2], hc_mult, hc_mult),
            dtype=pre.dtype,
            device=pre.device,
        )
    grad_x, grad_hc_fn, grad_scale, grad_base = tilelang_mhc_pre_bwd_op(
        grad_y,
        grad_post,
        grad_comb,
        x_float,
        rsqrt,
        mixes,
        pre,
        post,
        sinkhorn_norm,
        sinkhorn_sum,
        hc_fn,
        hc_scale,
        hc_base,
        ctx.eps,
    )
    return grad_x.to(ctx.input_dtype), grad_hc_fn, grad_scale, grad_base, None, None, None, None


def _tilelang_mhc_pre_fwd_setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
    _tilelang_mhc_pre_setup_context(ctx, inputs, output)
    ctx.mark_non_differentiable(*output[3:])


tilelang_mhc_pre_fwd_op.register_autograd(
    _tilelang_mhc_pre_backward,
    setup_context=_tilelang_mhc_pre_fwd_setup_context,
)


def tilelang_mhc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public HcPre wrapper returning only the model-visible outputs."""

    outputs = tilelang_mhc_pre_fwd_op(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps,
        norm_eps,
    )
    return outputs[0], outputs[1], outputs[2]
