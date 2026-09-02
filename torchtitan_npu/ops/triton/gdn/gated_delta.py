# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

# pylint: disable=huawei-too-many-arguments,huawei-too-many-return-values
_CHUNK_SIZE, _VALUE_BLOCK_SIZE = 64, 16
_GatedDeltaInputs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


@triton.jit
def _update_state(state, key, value, decay, beta):
    projected = decay * state
    residual = value - tl.sum(projected * key[:, None], axis=0)
    return projected + key[:, None] * (beta * residual)[None, :], residual


@triton.jit
def _state_gradients(adjoint, key, projected, residual, beta):
    grad_update = tl.sum(adjoint * key[:, None], axis=0)
    grad_residual = beta * grad_update
    grad_key = tl.sum(adjoint * (beta * residual)[None, :], axis=1)
    grad_key -= tl.sum(projected * grad_residual[None, :], axis=1)
    grad_projected = adjoint - key[:, None] * grad_residual[None, :]
    grad_beta = tl.sum(grad_update * residual, axis=0)
    return grad_residual, grad_key, grad_projected, grad_beta


@triton.jit
def _kernel_context(tokens, heads, dim, chunk_size, value_block_size):
    blocks, program = dim // value_block_size, tl.program_id(0)
    value_block, head_batch = program % blocks, program // blocks
    head, batch = head_batch % heads, head_batch // heads
    d = tl.arange(0, dim)
    vd = value_block * value_block_size + tl.arange(0, value_block_size)
    matrix = d[:, None] * dim + vd[None, :]
    checkpoint_base = head_batch * (tokens // chunk_size) * dim * dim
    return blocks, value_block, head_batch, head, batch, d, vd, matrix, checkpoint_base


@triton.jit
def _token_offset(batch, token, head, tokens, heads):
    return (batch * tokens + token) * heads + head, batch * tokens + token


@triton.jit
def _load_fp32_pair(first, second, first_offset, second_offset):
    return tl.load(first + first_offset).to(tl.float32), tl.load(second + second_offset).to(tl.float32)


@triton.jit
def _load_gate(decay, step_size, row):
    return tl.exp(tl.load(decay + row).to(tl.float32)), tl.load(step_size + row).to(tl.float32)


@triton.jit
def _gated_delta_forward_kernel(
    q,
    k,
    v,
    g,
    beta,
    reset,
    output,
    checkpoints,
    tokens,
    heads,
    scale: tl.constexpr,
    dim: tl.constexpr,
    chunk_size: tl.constexpr,
    value_block_size: tl.constexpr,
    store_output: tl.constexpr,
    store_checkpoints: tl.constexpr,
):
    _blocks, _value_block, _head_batch, head, batch, d, vd, matrix, checkpoint_base = _kernel_context(
        tokens, heads, dim, chunk_size, value_block_size
    )
    state = tl.zeros((dim, value_block_size), tl.float32)
    if store_checkpoints:
        tl.store(checkpoints + checkpoint_base + matrix, state)
    for start in range(0, tokens, chunk_size):
        for offset in range(chunk_size):
            token = start + offset
            row, reset_index = _token_offset(batch, token, head, tokens, heads)
            vector = row * dim
            kt, vt = _load_fp32_pair(k, v, vector + d, vector + vd)
            alpha, bt = _load_gate(g, beta, row)
            state = tl.where(tl.load(reset + reset_index) != 0, 0.0, state)
            state = _update_state(state, kt, vt, alpha, bt)[0]
            if store_output:
                qt = tl.load(q + vector + d).to(tl.float32)
                tl.store(output + vector + vd, scale * tl.sum(state * qt[:, None], axis=0))
        if store_checkpoints:
            checkpoint = start // chunk_size + 1
            address = checkpoints + checkpoint_base + checkpoint * dim * dim + matrix
            tl.store(address, state, mask=checkpoint < tokens // chunk_size)


@triton.jit
def _gated_delta_backward_kernel(
    q,
    k,
    v,
    g,
    beta,
    reset,
    dy,
    checkpoints,
    scratch,
    dqh,
    dkh,
    dv,
    da,
    db,
    tokens,
    heads,
    scale: tl.constexpr,
    dim: tl.constexpr,
    chunk_size: tl.constexpr,
    value_block_size: tl.constexpr,
):
    blocks, value_block, head_batch, head, batch, d, vd, matrix, checkpoint_base = _kernel_context(
        tokens, heads, dim, chunk_size, value_block_size
    )
    scratch_base = head_batch * (chunk_size + 1) * dim * dim
    adjoint_state = tl.zeros((dim, value_block_size), tl.float32)
    chunk = tokens // chunk_size - 1
    while chunk >= 0:
        start = chunk * chunk_size
        state = tl.load(checkpoints + checkpoint_base + chunk * dim * dim + matrix).to(tl.float32)
        tl.store(scratch + scratch_base + matrix, state)
        for offset in range(chunk_size):
            token = start + offset
            row, reset_index = _token_offset(batch, token, head, tokens, heads)
            vector = row * dim
            kt, vt = _load_fp32_pair(k, v, vector + d, vector + vd)
            alpha, bt = _load_gate(g, beta, row)
            state = tl.where(tl.load(reset + reset_index) != 0, 0.0, state)
            state = _update_state(state, kt, vt, alpha, bt)[0]
            tl.store(scratch + scratch_base + (offset + 1) * dim * dim + matrix, state)
        for reverse in range(chunk_size):
            offset = chunk_size - reverse - 1
            token = start + offset
            row, reset_index = _token_offset(batch, token, head, tokens, heads)
            vector = row * dim
            previous, current = _load_fp32_pair(
                scratch,
                scratch,
                scratch_base + offset * dim * dim + matrix,
                scratch_base + (offset + 1) * dim * dim + matrix,
            )
            qt, dyt = _load_fp32_pair(q, dy, vector + d, vector + vd)
            kt, vt = _load_fp32_pair(k, v, vector + d, vector + vd)
            alpha, bt = _load_gate(g, beta, row)
            reset_state = tl.load(reset + reset_index) != 0
            previous = tl.where(reset_state, 0.0, previous)
            projected = alpha * previous
            residual = vt - tl.sum(projected * kt[:, None], axis=0)
            tl.store(dqh + (row * blocks + value_block) * dim + d, scale * tl.sum(current * dyt[None, :], axis=1))
            adjoint = adjoint_state + scale * qt[:, None] * dyt[None, :]
            grad_residual, grad_k, grad_projected, grad_beta = _state_gradients(adjoint, kt, projected, residual, bt)
            tl.atomic_add(db + row, grad_beta)
            tl.store(dv + vector + vd, grad_residual)
            tl.atomic_add(dkh + vector + d, grad_k)
            tl.store(da + row * blocks + value_block, tl.sum(grad_projected * previous))
            adjoint_state = tl.where(reset_state, 0.0, alpha * grad_projected)
        chunk -= 1


def _l2_normalize(x: Tensor) -> tuple[Tensor, Tensor]:
    value = x.float()
    inverse = torch.rsqrt(torch.sum(value * value, dim=-1, keepdim=True) + 1e-6)
    return (value * inverse).to(x.dtype), inverse


def _run_forward_kernel(inputs: _GatedDeltaInputs, scale: float, *, save: bool):
    q, k, v, g, beta, reset = inputs
    batch, tokens, heads, dim = q.shape
    output = v if save else torch.empty_like(v)
    checkpoints = output
    if save:
        checkpoints = torch.empty(
            (batch, heads, tokens // _CHUNK_SIZE, dim, dim),
            dtype=torch.float32,
            device=q.device,
        )
    _gated_delta_forward_kernel[(batch * heads * (dim // _VALUE_BLOCK_SIZE),)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        reset=reset,
        output=output,
        checkpoints=checkpoints,
        tokens=tokens,
        heads=heads,
        scale=scale,
        dim=dim,
        chunk_size=_CHUNK_SIZE,
        value_block_size=_VALUE_BLOCK_SIZE,
        store_output=not save,
        store_checkpoints=save,
    )
    return output, checkpoints


def gated_delta_forward(inputs: _GatedDeltaInputs, scale: float):
    q, k, v, g, beta, reset = inputs
    qh, _ = _l2_normalize(q)
    kh, _ = _l2_normalize(k)
    return _run_forward_kernel((qh, kh, v, g, beta, reset), scale, save=False)[0]


def gated_delta_backward(inputs: _GatedDeltaInputs, dy: Tensor, scale: float):
    q, k, v, g, beta, reset = inputs
    batch, tokens, heads, dim = q.shape
    blocks = dim // _VALUE_BLOCK_SIZE
    qh, qi = _l2_normalize(q)
    kh, ki = _l2_normalize(k)
    _, checkpoints = _run_forward_kernel((qh, kh, v, g, beta, reset), scale, save=True)
    dqh = torch.empty((*q.shape[:3], blocks, dim), dtype=torch.float32, device=q.device)
    dkh = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    da = torch.empty((*g.shape, blocks), dtype=torch.float32, device=g.device)
    db = torch.zeros_like(beta, dtype=torch.float32)
    scratch = torch.empty(
        (batch, heads, _CHUNK_SIZE + 1, dim, dim),
        dtype=torch.float32,
        device=q.device,
    )
    _gated_delta_backward_kernel[(batch * heads * blocks,)](
        q=qh,
        k=kh,
        v=v,
        g=g,
        beta=beta,
        reset=reset,
        dy=dy,
        checkpoints=checkpoints,
        scratch=scratch,
        dqh=dqh,
        dkh=dkh,
        dv=dv,
        da=da,
        db=db,
        tokens=tokens,
        heads=heads,
        scale=scale,
        dim=dim,
        chunk_size=_CHUNK_SIZE,
        value_block_size=_VALUE_BLOCK_SIZE,
    )
    dqh = dqh.sum(3)
    qf, kf = q.float(), k.float()
    dq = dqh * qi - qf * torch.sum(qf * dqh, -1, True) * qi * qi * qi
    dk = dkh * ki - kf * torch.sum(kf * dkh, -1, True) * ki * ki * ki
    dg = da.sum(-1) * g.float().exp()
    dv = dv.to(v.dtype)
    return dq.to(q.dtype), dk.to(k.dtype), dv, dg.to(g.dtype), db.to(beta.dtype)


def _validate_inputs(inputs: _GatedDeltaInputs, scale: float) -> None:
    q, k, v, g, beta, reset = inputs
    if q.ndim != 4:
        raise ValueError("q must be a rank-4 tensor")
    if k.shape != q.shape:
        raise ValueError("k must match q in shape")
    if v.shape != q.shape:
        raise ValueError("v must match q in shape")
    if q.shape[-1] not in (64, 128) or q.shape[1] <= 0 or q.shape[1] % _CHUNK_SIZE:
        raise ValueError("head dim must be 64/128 and tokens a positive multiple of 64")
    if q.shape[0] <= 0 or q.shape[2] <= 0 or q.shape[0] * q.shape[2] > 8191:
        raise ValueError("batch and heads must be positive with product <= 8191")
    if q.dtype not in (torch.bfloat16, torch.float16) or not q.dtype == k.dtype == v.dtype:
        raise TypeError("q, k, and v must share the bfloat16 or float16 dtype")
    if any(x.dtype not in (q.dtype, torch.float32) for x in (g, beta)):
        raise TypeError("g and beta must match the input dtype or use float32")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must match the batch, token, and head dimensions")
    if any(x.device != q.device for x in (k, v, g, beta)):
        raise ValueError("all inputs must be on one device")
    if reset.shape != q.shape[:2] or reset.dtype != torch.bool or reset.device != q.device:
        raise ValueError("reset must be a boolean [batch, tokens] tensor on the input device")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")


@torch.library.custom_op("torchtitan_npu::gated_delta_rule", mutates_args=(), device_types="npu")
def gated_delta_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    reset: Tensor,
    *,
    scale: float,
) -> Tensor:
    inputs = q, k, v, g, beta, reset
    _validate_inputs(inputs, scale)
    return gated_delta_forward(inputs, scale)


@gated_delta_op.register_fake
def gated_delta_op_fake(q, k, v, g, beta, reset, *, scale):
    _validate_inputs((q, k, v, g, beta, reset), scale)
    return torch.empty_like(v, device=q.device)


@torch.library.custom_op("torchtitan_npu::gated_delta_rule_backward", mutates_args=(), device_types="npu")
def gated_delta_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    reset: Tensor,
    dy: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    inputs = q, k, v, g, beta, reset
    _validate_inputs(inputs, scale)
    if dy.shape != v.shape or dy.dtype != v.dtype or dy.device != v.device:
        raise ValueError("dy must match v in shape, dtype, and device")
    return gated_delta_backward(inputs, dy, scale)


@gated_delta_backward_op.register_fake
def gated_delta_backward_op_fake(q, k, v, g, beta, reset, dy, *, scale):
    _validate_inputs((q, k, v, g, beta, reset), scale)
    if dy.shape != v.shape or dy.dtype != v.dtype or dy.device != v.device:
        raise ValueError("dy must match v in shape, dtype, and device")
    return tuple(torch.empty_like(x) for x in (q, k, v, g, beta))


def gated_delta_setup_context(ctx, inputs, keyword_only_inputs, output):
    del output
    ctx.save_for_backward(*inputs)
    ctx.scale = keyword_only_inputs["scale"]


def gated_delta_autograd_backward(ctx, dy):
    q, k, v, g, beta, reset = ctx.saved_tensors
    grads = gated_delta_backward_op(
        q,
        k,
        v,
        g,
        beta,
        reset,
        dy.contiguous(),
        scale=ctx.scale,
    )
    return (*grads, None)


torch.library.register_autograd(
    gated_delta_op,
    gated_delta_autograd_backward,
    setup_context=gated_delta_setup_context,
)


def gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    reset: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    if q.shape[2] != v.shape[2]:
        if v.shape[2] % q.shape[2]:
            raise ValueError("Value heads must be a multiple of Q/K heads")
        repeat = v.shape[2] // q.shape[2]
        q, k = (x.repeat_interleave(repeat, 2) for x in (q, k))
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    reset = torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device) if reset is None else reset.contiguous()
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return gated_delta_op(q, k, v, g, beta, reset, scale=float(scale))
