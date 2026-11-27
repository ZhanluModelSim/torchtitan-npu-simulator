# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this tree.

"""Meta contract for the sys2mot mHC hyper-connection boundaries.

Three boundaries cover the reference's per-sublayer hyper-connection math:

- ``sys2mot_sim.mhc_mix`` folds the fp32 RMS-normed linear expansion, the
  sigmoid/pre and doubled-sigmoid/post gates, the row-softmax mixing matrix
  and the full Sinkhorn normalization loop into one boundary per sublayer.
  The per-iteration column/row renormalization with ``clamp_min`` guards is
  replicated exactly (iteration count carried as a schema scalar).
- ``sys2mot_sim.mhc_collapse`` folds the stream collapse
  ``(h * pre[..., None]).sum(-2)`` plus its dtype return cast.
- ``sys2mot_sim.mhc_expand`` folds the routed stream mixing einsum
  (``comb[i,j]`` routes input stream ``i`` into output stream ``j``) plus
  the gated residual injection and dtype return cast.

Each boundary pairs with a training-only backward boundary emitting the
gradients the reference autograd would produce (stream, projection weight,
base/scale gates; collapse stream/pre; expand stream/matrix/post/y).

The operators declare capture boundaries, not executable NPU kernels; real
device calls fail explicitly (the compat install only happens inside Meta
capture sessions).
"""

import torch


@torch.library.custom_op("sys2mot_sim::mhc_mix", mutates_args=())
def mhc_mix(
    stream: torch.Tensor,
    weight: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError("sys2mot virtual mhc_mix is available only for Meta simulation")


@mhc_mix.register_fake
def _mhc_mix_meta(stream, weight, base, scale, eps, sinkhorn_iters):
    if stream.ndim != 4 or min(stream.shape) <= 0:
        raise ValueError("mhc mix stream must be a positive (batch, seq, n, width) block")
    batch, seq, n, width = stream.shape
    if stream.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("mhc mix reads the model stream dtype")
    out_features = (n + 2) * n
    if weight.shape != (out_features, n * width) or weight.dtype != torch.float32:
        raise ValueError("mhc mix expansion must emit (n+2)*n mixing values in float32")
    if base.shape != (out_features,) or base.dtype != torch.float32:
        raise ValueError("mhc mix base must be one float32 bias per mixing value")
    if scale.shape != (3,) or scale.dtype != torch.float32:
        raise ValueError("mhc mix scale must be three float32 gate weights")
    if eps <= 0 or sinkhorn_iters < 0:
        raise ValueError("mhc mix epsilon must be positive and iterations nonnegative")
    pre = torch.empty((batch, seq, n), dtype=torch.float32, device=stream.device)
    post = torch.empty((batch, seq, n), dtype=torch.float32, device=stream.device)
    matrix = torch.empty((batch, seq, n, n), dtype=torch.float32, device=stream.device)
    return pre, post, matrix


@torch.library.custom_op("sys2mot_sim::mhc_mix_backward", mutates_args=())
def mhc_mix_backward(
    stream: torch.Tensor,
    weight: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    upstream_pre: torch.Tensor,
    upstream_post: torch.Tensor,
    upstream_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError(
        "sys2mot virtual mhc_mix backward is available only for Meta simulation"
    )


@mhc_mix_backward.register_fake
def _mhc_mix_backward_meta(stream, weight, base, scale, upstream_pre, upstream_post, upstream_matrix):
    batch, seq, n = stream.shape[:3]
    if upstream_pre.shape != (batch, seq, n) or upstream_post.shape != (batch, seq, n):
        raise ValueError("mhc mix backward gate gradients must match the stream grid")
    if upstream_matrix.shape != (batch, seq, n, n):
        raise ValueError("mhc mix backward matrix gradient must match the mixing matrix")
    return (
        torch.empty_like(stream),
        torch.empty_like(weight),
        torch.empty_like(base),
        torch.empty_like(scale),
    )


def _mix_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3])


def _mix_backward(ctx, grad_pre, grad_post, grad_matrix):
    stream, weight, base, scale = ctx.saved_tensors
    grads = mhc_mix_backward(stream, weight, base, scale, grad_pre, grad_post, grad_matrix)
    return (*grads, None, None)


mhc_mix.register_autograd(_mix_backward, setup_context=_mix_setup)


@torch.library.custom_op("sys2mot_sim::mhc_collapse", mutates_args=())
def mhc_collapse(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("sys2mot virtual mhc_collapse is available only for Meta simulation")


@mhc_collapse.register_fake
def _mhc_collapse_meta(stream, pre):
    # The reference collapse promotes the product to fp32 through h.float(),
    # so pre may arrive as either the fp32 mix gate or the bf16 identity gate.
    if stream.ndim != 4 or pre.shape != stream.shape[:3]:
        raise ValueError("mhc collapse pre must gate every stream row")
    if pre.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("mhc collapse pre must be a floating gate")
    return torch.empty(stream.shape[:2] + stream.shape[3:], dtype=stream.dtype, device=stream.device)


@torch.library.custom_op("sys2mot_sim::mhc_collapse_backward", mutates_args=())
def mhc_collapse_backward(
    stream: torch.Tensor, pre: torch.Tensor, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    raise RuntimeError(
        "sys2mot virtual mhc_collapse backward is available only for Meta simulation"
    )


@mhc_collapse_backward.register_fake
def _mhc_collapse_backward_meta(stream, pre, upstream):
    if upstream.shape != stream.shape[:2] + stream.shape[3:]:
        raise ValueError("mhc collapse backward upstream must match the collapsed rows")
    return torch.empty_like(stream), torch.empty_like(pre)


def _collapse_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _collapse_backward(ctx, grad):
    stream, pre = ctx.saved_tensors
    stream_grad, pre_grad = mhc_collapse_backward(stream, pre, grad)
    return stream_grad, pre_grad


mhc_collapse.register_autograd(_collapse_backward, setup_context=_collapse_setup)


@torch.library.custom_op("sys2mot_sim::mhc_expand", mutates_args=())
def mhc_expand(
    stream: torch.Tensor, matrix: torch.Tensor, post: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    raise RuntimeError("sys2mot virtual mhc_expand is available only for Meta simulation")


@mhc_expand.register_fake
def _mhc_expand_meta(stream, matrix, post, y):
    batch, seq, n, width = stream.shape
    if matrix.shape != (batch, seq, n, n) or matrix.dtype != torch.float32:
        raise ValueError("mhc expand matrix must route n streams per row in float32")
    if post.shape != (batch, seq, n) or post.dtype != torch.float32:
        raise ValueError("mhc expand post must gate every output stream in float32")
    if y.shape != (batch, seq, width) or y.dtype != stream.dtype:
        raise ValueError("mhc expand residual must match the collapsed row width")
    return torch.empty_like(stream)


@torch.library.custom_op("sys2mot_sim::mhc_expand_backward", mutates_args=())
def mhc_expand_backward(
    stream: torch.Tensor,
    matrix: torch.Tensor,
    post: torch.Tensor,
    y: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError(
        "sys2mot virtual mhc_expand backward is available only for Meta simulation"
    )


@mhc_expand_backward.register_fake
def _mhc_expand_backward_meta(stream, matrix, post, y, upstream):
    if upstream.shape != stream.shape:
        raise ValueError("mhc expand backward upstream must match the expanded streams")
    return (
        torch.empty_like(stream),
        torch.empty_like(matrix),
        torch.empty_like(post),
        torch.empty_like(y),
    )


def _expand_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3])


def _expand_backward(ctx, grad):
    stream, matrix, post, y = ctx.saved_tensors
    return mhc_expand_backward(stream, matrix, post, y, grad)


mhc_expand.register_autograd(_expand_backward, setup_context=_expand_setup)


def install_sys2mot_mhc_meta_support(scope) -> dict[str, str]:
    """Route the sys2mot mHC hyper-connections through the fused boundaries."""
    import torchtitan_npu.models.sys2mot.model as sys2mot_model

    def mix_forward(self, h: torch.Tensor):
        return mhc_mix(
            h,
            self.fn.weight.float(),
            self.base.float(),
            self.scale.float(),
            self.eps,
            self.sinkhorn_iters,
        )

    def collapse_forward(h: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return mhc_collapse(h, pre)

    def expand_forward(
        y: torch.Tensor, h: torch.Tensor, post: torch.Tensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        return mhc_expand(h, matrix, post, y)

    scope.replace(sys2mot_model.HyperMix, "forward", mix_forward)
    scope.replace(sys2mot_model.HyperMix, "collapse", collapse_forward)
    scope.replace(sys2mot_model.HyperMix, "expand", expand_forward)
    return {
        "mhc_mix_capture_boundary": "sys2mot_sim.mhc_mix",
        "mhc_collapse_capture_boundary": "sys2mot_sim.mhc_collapse",
        "mhc_expand_capture_boundary": "sys2mot_sim.mhc_expand",
    }
