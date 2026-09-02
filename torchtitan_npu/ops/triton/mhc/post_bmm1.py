# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_hc_post_bmm1_fwd_kernel(
    H_post_ptr,
    h_out_ptr,
    h_post_ptr,
    BS,
    C,
    stride_hp_bs: tl.constexpr,
    stride_hp_n: tl.constexpr,
    stride_ho_bs: tl.constexpr,
    stride_ho_c: tl.constexpr,
    stride_y_bs: tl.constexpr,
    stride_y_n: tl.constexpr,
    stride_y_c: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DIVISIBLE_C: tl.constexpr,
):
    pid_bs_blk = tl.program_id(0)
    pid_c_blk = tl.program_id(1)

    # bs tile
    pid0 = pid_bs_blk * GROUP
    pids = pid0 + tl.arange(0, GROUP)
    mask_pid = pids < BS

    # c tile
    c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = tl.full((BLOCK_C,), True, tl.int1) if DIVISIBLE_C else (c < C)

    m = mask_pid[:, None] & mask_c[None, :]

    # load h_out once (G,BC)
    hout = tl.load(
        h_out_ptr + pids[:, None] * stride_ho_bs + c[None, :] * stride_ho_c,
        mask=m,
        other=0.0,
    ).to(tl.float32)

    # load H_post scalars (G,1)
    hp0 = tl.load(H_post_ptr + pids * stride_hp_bs + 0 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp1 = tl.load(H_post_ptr + pids * stride_hp_bs + 1 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp2 = tl.load(H_post_ptr + pids * stride_hp_bs + 2 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp3 = tl.load(H_post_ptr + pids * stride_hp_bs + 3 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]

    # store h_post[bs, i, c]
    Y_base = h_post_ptr + pids[:, None] * stride_y_bs + c[None, :] * stride_y_c
    tl.store(Y_base + 0 * stride_y_n, hout * hp0, mask=m)
    tl.store(Y_base + 1 * stride_y_n, hout * hp1, mask=m)
    tl.store(Y_base + 2 * stride_y_n, hout * hp2, mask=m)
    tl.store(Y_base + 3 * stride_y_n, hout * hp3, mask=m)


@triton.jit
def _triton_hc_post_bmm1_bwd_fused_kernel(
    H_post_ptr,
    h_out_ptr,
    dY_ptr,
    dh_out_ptr,
    dH_post_ptr,
    BS,
    C,
    stride_hp_bs: tl.constexpr,
    stride_hp_n: tl.constexpr,
    stride_ho_bs: tl.constexpr,
    stride_ho_c: tl.constexpr,
    stride_dy_bs: tl.constexpr,
    stride_dy_n: tl.constexpr,
    stride_dy_c: tl.constexpr,
    stride_dho_bs: tl.constexpr,
    stride_dho_c: tl.constexpr,
    stride_dhp_bs: tl.constexpr,
    stride_dhp_n: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DIVISIBLE_C: tl.constexpr,
):
    pid_bs_blk = tl.program_id(0)
    pid_c_blk = tl.program_id(1)

    # bs tile
    pid0 = pid_bs_blk * GROUP
    pids = pid0 + tl.arange(0, GROUP)  # (G,)
    mask_pid = pids < BS

    # c tile
    c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)  # (BC,)
    mask_c = tl.full((BLOCK_C,), True, tl.int1) if DIVISIBLE_C else (c < C)

    m = mask_pid[:, None] & mask_c[None, :]

    # load h_out (G,BC)
    hout = tl.load(
        h_out_ptr + pids[:, None] * stride_ho_bs + c[None, :] * stride_ho_c,
        mask=m,
        other=0.0,
    ).to(tl.float32)

    # load H_post scalars (G,1)
    hp0 = tl.load(H_post_ptr + pids * stride_hp_bs + 0 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp1 = tl.load(H_post_ptr + pids * stride_hp_bs + 1 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp2 = tl.load(H_post_ptr + pids * stride_hp_bs + 2 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp3 = tl.load(H_post_ptr + pids * stride_hp_bs + 3 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]

    # dh_out accumulator (G,BC)
    dh = tl.zeros((GROUP, BLOCK_C), dtype=tl.float32)

    # base pointer for dY (G,BC) per i
    dY_base = dY_ptr + pids[:, None] * stride_dy_bs + c[None, :] * stride_dy_c

    dy = tl.load(dY_base + 0 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
    dh += dy * hp0
    dhp = tl.sum(dy * hout, axis=1)  # (G,)
    tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 0 * stride_dhp_n, dhp, mask=mask_pid)

    dy = tl.load(dY_base + 1 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
    dh += dy * hp1
    dhp = tl.sum(dy * hout, axis=1)
    tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 1 * stride_dhp_n, dhp, mask=mask_pid)

    dy = tl.load(dY_base + 2 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
    dh += dy * hp2
    dhp = tl.sum(dy * hout, axis=1)
    tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 2 * stride_dhp_n, dhp, mask=mask_pid)

    dy = tl.load(dY_base + 3 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
    dh += dy * hp3
    dhp = tl.sum(dy * hout, axis=1)
    tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 3 * stride_dhp_n, dhp, mask=mask_pid)

    # store dh_out (G,BC)
    tl.store(
        dh_out_ptr + pids[:, None] * stride_dho_bs + c[None, :] * stride_dho_c,
        dh,
        mask=m,
    )


def hc_post_bmm1_forward(h_out: torch.Tensor, H_post: torch.Tensor) -> torch.Tensor:
    """
    h_out : [B,S,C] bf16
    H_post: [B,S,4] fp32
    out   : [B,S,4,C] fp32
    """
    if h_out.ndim != 3 or H_post.ndim != 3:
        raise ValueError("input in hc_post_bmm1_forward dim error")
    B, S, C = h_out.shape
    B2, S2, N = H_post.shape
    if (B, S) != (B2, S2):
        raise ValueError("input in hc_post_bmm1_forward shape error")
    if N != 4:
        raise ValueError("input in hc_post_bmm1_forward shape error")

    BS = B * S

    GROUP = 2
    BLOCK_C = C

    DIV_C = C % BLOCK_C == 0

    HO = h_out.contiguous().view(BS, C)
    HP = H_post.contiguous().view(BS, N)  # fp32
    Y = torch.empty((BS, N, C), device=h_out.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(C, BLOCK_C))
    _triton_hc_post_bmm1_fwd_kernel[grid](
        HP,
        HO,
        Y,
        BS,
        C,
        stride_hp_bs=HP.stride(0),
        stride_hp_n=HP.stride(1),
        stride_ho_bs=HO.stride(0),
        stride_ho_c=HO.stride(1),
        stride_y_bs=Y.stride(0),
        stride_y_n=Y.stride(1),
        stride_y_c=Y.stride(2),
        GROUP=GROUP,
        BLOCK_C=BLOCK_C,
        DIVISIBLE_C=DIV_C,
    )

    return Y.view(B, S, N, C)


def hc_post_bmm1_backward(h_out: torch.Tensor, H_post: torch.Tensor, grad_out: torch.Tensor):
    """
    h_out   : [B,S,C] bf16
    H_post  : [B,S,4] fp32
    grad_out: [B,S,4,C] fp32 (or castable)
    returns:
      grad_h_out : [B,S,C] fp32
      grad_H_post: [B,S,4] fp32
    """
    if h_out.ndim != 3 or H_post.ndim != 3 or grad_out.ndim != 4:
        raise ValueError("input in hc_post_bmm1_backward dim error")

    B, S, C = h_out.shape
    B2, S2, N = H_post.shape
    B3, S3, N3, C3 = grad_out.shape
    if (B, S, C) != (B3, S3, C3) or (B, S, N) != (B2, S2, N3):
        raise ValueError("input in hc_post_bmm1_backward shape error")

    if N != 4:
        raise ValueError("input in hc_post_bmm1_backward shape error")

    BS = B * S

    GROUP = 2
    if C > 4096 and C % 2 != 0:
        raise ValueError(f"Channel dimension C must be even to split into two equal blocks, but got {C}")

    BLOCK_C = C // 2 if C > 4096 else C

    DIV_C = C % BLOCK_C == 0

    HO = h_out.contiguous().view(BS, C)
    HP = H_post.contiguous().view(BS, N)  # fp32
    dY = grad_out.contiguous().view(BS, N, C).to(torch.float32)

    dHO = torch.empty((BS, C), device=h_out.device, dtype=torch.float32)
    dHP = torch.zeros((BS, N), device=h_out.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(C, BLOCK_C))
    _triton_hc_post_bmm1_bwd_fused_kernel[grid](
        HP,
        HO,
        dY,
        dHO,
        dHP,
        BS,
        C,
        stride_hp_bs=HP.stride(0),
        stride_hp_n=HP.stride(1),
        stride_ho_bs=HO.stride(0),
        stride_ho_c=HO.stride(1),
        stride_dy_bs=dY.stride(0),
        stride_dy_n=dY.stride(1),
        stride_dy_c=dY.stride(2),
        stride_dho_bs=dHO.stride(0),
        stride_dho_c=dHO.stride(1),
        stride_dhp_bs=dHP.stride(0),
        stride_dhp_n=dHP.stride(1),
        GROUP=GROUP,
        BLOCK_C=BLOCK_C,
        DIVISIBLE_C=DIV_C,
    )

    return dHO.view(B, S, C), dHP.view(B, S, N)


@torch.library.custom_op("torchtitan_npu::mhc_post_bmm1", mutates_args=(), device_types="npu")
def mhc_post_bmm1_op(h_out: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V4 mHC post bmm1: scale h_out by the per-stream h_post weights."""
    return hc_post_bmm1_forward(h_out, h_post)


@mhc_post_bmm1_op.register_fake
def _mhc_post_bmm1_fake(h_out, h_post):
    if h_out.ndim != 3 or h_post.ndim != 3:
        raise ValueError("shape error in mhc_post_bmm1")
    b, s, c = h_out.shape
    n = h_post.shape[-1]
    return torch.empty((b, s, n, c), device=h_out.device, dtype=torch.float32)


@torch.library.custom_op("torchtitan_npu::mhc_post_bmm1_bwd", mutates_args=(), device_types="npu")
def mhc_post_bmm1_bwd_op(
    h_out: torch.Tensor, h_post: torch.Tensor, grad_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile-safe backward wrapper around the raw Triton kernel launch."""
    return hc_post_bmm1_backward(h_out, h_post, grad_out)


@mhc_post_bmm1_bwd_op.register_fake
def _mhc_post_bmm1_bwd_fake(h_out, h_post, grad_out):
    if h_out.ndim != 3 or h_post.ndim != 3:
        raise ValueError("shape error in mhc_post_bmm1_bwd")
    b, s, c = h_out.shape
    n = h_post.shape[-1]
    return (
        torch.empty((b, s, c), device=h_out.device, dtype=torch.float32),
        torch.empty((b, s, n), device=h_out.device, dtype=torch.float32),
    )


def _mhc_post_bmm1_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _mhc_post_bmm1_backward(ctx, grad_out):
    h_out, h_post = ctx.saved_tensors
    return mhc_post_bmm1_bwd_op(h_out, h_post, grad_out)


torch.library.register_autograd(
    mhc_post_bmm1_op,
    _mhc_post_bmm1_backward,
    setup_context=_mhc_post_bmm1_setup_context,
)
