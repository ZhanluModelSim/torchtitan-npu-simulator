# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview multi-hyper-connect (MHC) stream-mixing primitives.

Fork reason: MAGI-2-preview replaces residual connections with MHC stream
mixing; upstream torchtitan has no equivalent.
Reference: inference/model/magi2_preview.py (_sigmoid_affine, _sinkhorn_knopp,
_apply_hpre, _hyper_connect torch paths; Triton kernels are inference-only).

All math runs in fp32. ``alpha``/``bias`` are the layer-owned MHC parameters
and ``matmul_scale`` is ``1 / sqrt(num_stream * hidden_size)``.
"""

import torch


def sigmoid_affine(
    x: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    matmul_scale: float,
    sigmoid_scale: float = 1.0,
) -> torch.Tensor:
    """Scaled sigmoid gate ``sigmoid_scale * sigmoid(alpha * scale * x + bias)``.

    Used for the pre-mix coefficients (``sigmoid_scale=1.0``) and post-mix
    coefficients (``sigmoid_scale=2.0``). Broadcasts ``bias`` over ``x``.
    """
    return sigmoid_scale * torch.sigmoid(alpha * matmul_scale * x.float() + bias.float())


def sinkhorn_knopp(h: torch.Tensor, num_iters: int = 20, eps: float = 1e-12) -> torch.Tensor:
    """Sinkhorn-Knopp normalization towards a doubly stochastic matrix.

    Matches the reference: subtract the max over the last two dims for
    numerical stability, exponentiate, then alternate column and row
    normalization ``num_iters`` times.
    """
    h = h.float()
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


def apply_hpre(h_pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mix the multi-stream state into the sublayer input.

    Args:
        h_pre: pre-mix coefficients ``(T, n)``.
        x: stream state ``(T, n, C)``.

    Returns:
        Sublayer input ``(T, C)``.
    """
    return torch.einsum("tn,tnc->tc", h_pre, x)


def hyper_connect(
    res: torch.Tensor,
    out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    """Update the stream state after a sublayer.

    Args:
        res: previous stream state ``(T, n, C)``.
        out: sublayer output ``(T, C)``.
        h_post: post-mix coefficients ``(T, n)``.
        h_res: Sinkhorn-normalized residual mixing matrix ``(T, n, n)``.

    Returns:
        New stream state ``(T, n, C)``.
    """
    out_mstream = torch.einsum("tn,tc->tnc", h_post, out)
    mixed_res = torch.einsum("tij,tjc->tic", h_res, res)
    return mixed_res + out_mstream
