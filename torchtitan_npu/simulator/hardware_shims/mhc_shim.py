# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim for MHC's HcPre (`torchtitan_npu.converters.kernels.mhc_prepost.NpuHcPre` /
`torchtitan_npu.ops.triton.mhc_triton.MHCPreTriton` in production). Records the real production
op names (`npu_rms_norm`, `matmul`, and the Triton kernels `hc_pre_fwd`/`hc_pre_bmm_forward`, or
their backward counterparts) into the active OpDispatchCapture, with analytically-derived shapes
-- never invoking torch_npu or Triton for real. See design doc for the exact shape formulas."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


def _empty_like_shape(shape: tuple[int, ...], ref: torch.Tensor) -> torch.Tensor:
    return torch.empty(shape, dtype=ref.dtype, device=ref.device)


class _SimHcPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hc_fn, hc_scale, hc_base, hc_mult, module_path):  # noqa: ANN001
        B, S, nD = x.shape
        D = nD // hc_mult
        total = hc_fn.shape[0]
        dtype = x.dtype

        x_flat = x.reshape(B * S, nD)
        x_norm_flat = _empty_like_shape((B * S, nD), x_flat)
        rstd = _empty_like_shape((B * S, 1), x_flat)
        _record("npu.npu_rms_norm.default", [x_flat], [x_norm_flat, rstd], module_path)

        x_norm_mat = x_norm_flat.reshape(B, S, nD)
        x_proj = _empty_like_shape((B, S, total), x_flat)
        _record("aten.matmul.default", [x_norm_mat, hc_fn], [x_proj], module_path)

        h_pre = _empty_like_shape((B, S, hc_mult), x_flat)
        h_post = _empty_like_shape((B, S, hc_mult), x_flat)
        h_res = _empty_like_shape((B, S, hc_mult, hc_mult), x_flat)
        _record("triton.hc_pre_fwd", [x_proj, hc_scale, hc_base], [h_pre, h_post, h_res], module_path)

        x_unflatten = x.unflatten(dim=-1, sizes=(hc_mult, D))
        y = torch.empty((B, S, D), dtype=dtype, device=x.device)
        _record("triton.hc_pre_bmm_forward", [h_pre, x_unflatten], [y], module_path)

        ctx.save_for_backward(x, hc_fn, hc_scale, hc_base, h_pre)
        ctx.hc_mult, ctx.module_path = hc_mult, module_path
        ctx.B, ctx.S, ctx.nD, ctx.D, ctx.total = B, S, nD, D, total
        return y, h_post, h_res

    @staticmethod
    def backward(ctx, grad_y, grad_h_post, grad_h_res):  # noqa: ANN001
        x, hc_fn, hc_scale, hc_base, h_pre = ctx.saved_tensors
        B, S, nD, D, hc_mult, total = ctx.B, ctx.S, ctx.nD, ctx.D, ctx.hc_mult, ctx.total

        x_unflatten_shape = (B, S, hc_mult, D)
        grad_h_pre = _empty_like_shape((B, S, hc_mult), h_pre)
        grad_x_direct = torch.empty(x_unflatten_shape, dtype=x.dtype, device=x.device)
        _record(
            "triton.hc_pre_bmm_backward",
            [h_pre, x.unflatten(dim=-1, sizes=(hc_mult, D)), grad_y],
            [grad_h_pre, grad_x_direct],
            ctx.module_path,
        )

        grad_x_proj = _empty_like_shape((B, S, total), h_pre)
        grad_branch_alpha = torch.empty_like(hc_scale)
        grad_branch_beta = torch.empty_like(hc_base)
        _record(
            "triton.hc_pre_bwd",
            [grad_h_pre, grad_h_post, grad_h_res, x, hc_scale, hc_base],
            [grad_x_proj, grad_branch_alpha, grad_branch_beta],
            ctx.module_path,
        )

        grad_x = torch.empty((B, S, nD), dtype=x.dtype, device=x.device)
        grad_hc_fn = torch.empty_like(hc_fn)
        return grad_x, grad_hc_fn, grad_branch_alpha, grad_branch_beta, None, None


class SimHcPre(nn.Module):
    """Drop-in simulator replacement for `NpuHcPre`/`NpuHcPreFused`
    (`torchtitan_npu.converters.kernels.mhc_prepost`). Same forward()
    signature; never runs real Triton/torch_npu, only records the real op
    names + analytically-correct shapes.

    `__init__` mirrors `NpuHcPre.__init__(self, parent: HcPre)` exactly
    (shallow `__dict__` copy, no `super().__init__()` call): `parent` is
    already a fully-initialized `nn.Module` instance, so its `__dict__`
    already carries every internal `nn.Module` bookkeeping attribute
    (`_parameters`, `_buffers`, `_modules`, hook registries, ...) alongside
    `hc_mult`/`hc_sinkhorn_iters`/`hc_eps`/`norm_eps` -- copying it in one
    step reproduces both correctly without hardcoding `HcPre.Config`'s
    field names here."""

    def __init__(self, parent: "HcPre") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        x = x.flatten(2)
        module_path = self.__class__.__name__
        return _SimHcPreFn.apply(x, hc_fn, hc_scale, hc_base, self.hc_mult, module_path)
