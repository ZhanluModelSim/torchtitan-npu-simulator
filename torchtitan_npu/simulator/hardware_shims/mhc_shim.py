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

from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
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


class SimHcPre(HcPre):
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

    def __init__(self, parent: HcPre) -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        x = x.flatten(2)
        module_path = self.__class__.__name__
        return _SimHcPreFn.apply(x, hc_fn, hc_scale, hc_base, self.hc_mult, module_path)


class _SimHcHeadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hc_head_fn, hc_head_scale, hc_head_base, hc_mult, module_path):  # noqa: ANN001
        # x arrives pre-flattened to [B,S,nD] by SimHcHead.forward, matching NpuHcHead.
        B, S, nD = x.shape
        D = nD // hc_mult
        dtype = x.dtype

        # NpuHcHead's real forward calls MHCPreOnlyTriton.apply(x, hc_head_fn, hc_head_scale,
        # hc_head_base, ...) directly on the already-flattened x. Tracing MHCPreOnlyTriton.forward
        # (ops/triton/mhc_triton.py): `weight = hc_head_fn.t()` transposes hc_head_fn's real shape
        # `[hc_mult, hc_dim]` (HcHead.__init__, model.py:1064-1084) to `[hc_dim, hc_mult]`, so
        # `x_proj = matmul(x_norm_mat[B,S,nD], weight[nD,hc_mult]) = [B,S,hc_mult]` -- the "mixes"
        # last dim here is `hc_mult`, NOT `n*n+2*n` (that larger size is specific to the main
        # HcPre/MHCPreTriton path in SimHcPre, which uses a differently-shaped hc_fn weight).
        # hc_head_scale/hc_head_base are real HcHead parameters too: shape [1] and [hc_mult]
        # respectively (NOT [3]/[n*n+2*n] like HcPre's hc_scale/hc_base) -- confirmed from
        # HcHead.__init__'s actual nn.Parameter allocations.
        mixes = _empty_like_shape((B, S, hc_mult), x)
        h_pre = _empty_like_shape((B, S, hc_mult), x)
        _record("triton.hc_pre_only_fwd", [mixes, hc_head_scale, hc_head_base], [h_pre], module_path)

        x_unflatten = x.unflatten(dim=-1, sizes=(hc_mult, D))
        y = torch.empty((B, S, D), dtype=dtype, device=x.device)
        _record("triton.hc_pre_bmm_forward", [h_pre, x_unflatten], [y], module_path)

        ctx.save_for_backward(x, hc_head_fn, hc_head_scale, hc_head_base, h_pre)
        ctx.hc_mult, ctx.module_path = hc_mult, module_path
        ctx.B, ctx.S, ctx.nD, ctx.D = B, S, nD, D
        return y

    @staticmethod
    def backward(ctx, grad_y):  # noqa: ANN001
        x, hc_head_fn, hc_head_scale, hc_head_base, h_pre = ctx.saved_tensors
        B, S, nD, D, hc_mult = ctx.B, ctx.S, ctx.nD, ctx.D, ctx.hc_mult

        grad_h_pre = _empty_like_shape((B, S, hc_mult), h_pre)
        grad_x_direct = torch.empty((B, S, hc_mult, D), dtype=x.dtype, device=x.device)
        _record(
            "triton.hc_pre_bmm_backward",
            [h_pre, x.unflatten(dim=-1, sizes=(hc_mult, D)), grad_y],
            [grad_h_pre, grad_x_direct],
            ctx.module_path,
        )

        grad_x_proj = _empty_like_shape((B, S, hc_mult), h_pre)
        grad_branch_alpha = torch.empty_like(hc_head_scale)
        grad_branch_beta = torch.empty_like(hc_head_base)
        _record(
            "triton.hc_pre_only_bwd",
            [grad_h_pre, x, hc_head_scale, hc_head_base],
            [grad_x_proj, grad_branch_alpha, grad_branch_beta],
            ctx.module_path,
        )

        grad_x = torch.empty((B, S, nD), dtype=x.dtype, device=x.device)
        grad_hc_head_fn = torch.empty_like(hc_head_fn)
        return grad_x, grad_hc_head_fn, grad_branch_alpha, grad_branch_beta, None, None


class SimHcHead(HcHead):
    """Drop-in simulator replacement for `NpuHcHead`
    (`torchtitan_npu.converters.kernels.mhc_prepost`).

    `__init__` mirrors `NpuHcHead.__init__(self, parent: HcHead)`'s exact
    `__dict__` shallow-copy pattern -- **required**, not just style: unlike
    `HcPre`/`HcPost`, `HcHead` owns real `nn.Parameter`s
    (`hc_head_fn: [hc_mult,hc_dim]`, `hc_head_base: [hc_mult]`,
    `hc_head_scale: [1]`, per `HcHead.__init__` in
    `torchtitan_npu/models/deepseek_v4/model.py:1064-1084`) that must be
    reused verbatim from `parent`, not recreated with a guessed shape.

    **Important, found via real-container validation (Task 7):** unlike
    `HcPre`, `HcHead` does NOT store `hc_mult` as an instance attribute --
    `HcHead.__init__` only uses it as a local variable to size
    `hc_head_fn`/`hc_head_base` (`hc_dim = hc_mult * config.dim`), never
    doing `self.hc_mult = ...`. So `self.__dict__.update(parent.__dict__)`
    correctly carries over `hc_head_fn`/`hc_head_base`/`hc_head_scale`/
    `hc_eps`/`norm_eps`, but there is no `self.hc_mult` to copy -- `forward`
    must derive it from `self.hc_head_fn.shape[0]` (`hc_head_fn`'s real
    shape is `[hc_mult, hc_dim]`), not reference a nonexistent
    `self.hc_mult`."""

    def __init__(self, parent: HcHead) -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor):
        x = x.flatten(2)
        module_path = self.__class__.__name__
        hc_mult = self.hc_head_fn.shape[0]
        return _SimHcHeadFn.apply(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, hc_mult, module_path)


class _SimHcPostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, h_post, h_res, module_path):  # noqa: ANN001
        B, S, D = x.shape
        N = h_post.shape[-1]
        dtype = x.dtype

        bmm1 = torch.empty((B, S, N, D), dtype=torch.float32, device=x.device)
        _record("triton.hc_post_bmm1_forward", [x, h_post], [bmm1], module_path)

        residual_unflat = residual.view(B, S, N, D)
        bmm2 = torch.empty((B, S, N, D), dtype=torch.float32, device=x.device)
        _record("triton.hc_post_bmm2_forward", [h_res, residual_unflat], [bmm2], module_path)

        result_flat = torch.empty((B * S, N * D), dtype=torch.float32, device=x.device)
        _record(
            "triton.add_fwd",
            [bmm1.reshape(B * S, -1), bmm2.reshape(B * S, -1)],
            [result_flat],
            module_path,
        )

        ctx.save_for_backward(x, residual, h_post, h_res)
        ctx.module_path = module_path
        ctx.B, ctx.S, ctx.D, ctx.N = B, S, D, N
        return result_flat.view(B, S, N * D).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        x, residual, h_post, h_res = ctx.saved_tensors
        B, S, D, N = ctx.B, ctx.S, ctx.D, ctx.N

        grad_out_4d = grad_output.view(B, S, N, D).float()

        grad_x = torch.empty((B, S, D), dtype=torch.float32, device=x.device)
        grad_h_post = torch.empty((B, S, N), dtype=torch.float32, device=x.device)
        _record("triton.hc_post_bmm1_backward", [x, h_post, grad_out_4d], [grad_x, grad_h_post], ctx.module_path)

        residual_unflat = residual.view(B, S, N, D)
        grad_h_res = torch.empty((B, S, N, N), dtype=torch.float32, device=x.device)
        grad_residual_unflat = torch.empty((B, S, N, D), dtype=torch.float32, device=x.device)
        _record(
            "triton.hc_post_bmm2_backward",
            [h_res, residual_unflat, grad_out_4d],
            [grad_h_res, grad_residual_unflat],
            ctx.module_path,
        )
        grad_residual = grad_residual_unflat.flatten(-2)

        return grad_x, grad_residual, grad_h_post, grad_h_res, None


class SimHcPost(HcPost):
    """Drop-in simulator replacement for `NpuHcPost`
    (`torchtitan_npu.converters.kernels.mhc_prepost`). `__init__` mirrors
    `NpuHcPost.__init__(self, parent: HcPost)`'s `__dict__` shallow-copy
    pattern for consistency with `SimHcPre`/`SimHcHead`, even though
    `HcPost` owns no extra parameters beyond base `Module`.

    `forward` mirrors `NpuHcPost.forward`'s exact wrapping (mhc_prepost.py:236-278),
    which is a THIN WRAPPER around `MHCPostTriton.apply(...)` (mirrored here by
    `_SimHcPostFn`): it takes `residual` as 4D `[B,S,N,D]`, flattens it to
    `[B,S,N*D]` before calling the lower-level function (matching
    `_SimHcPostFn`'s own `residual.view(B,S,N,D)` internal-unflatten
    convention -- `_SimHcPostFn` expects an already-flattened `residual`,
    exactly like `MHCPostTriton.forward` does), then reshapes the `[B,S,N*D]`
    result back to 4D `[B,S,N,D]` before returning -- `NpuHcPost.forward`
    does the identical `y = y.view(dim_b, dim_s, dim_n, dim_d)` reshape
    at its very end (mhc_prepost.py:277) before returning to its caller."""

    def __init__(self, parent: HcPost) -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        dim_b, dim_s, dim_n, dim_d = residual.shape
        residual_flat = residual.flatten(2)
        module_path = self.__class__.__name__
        y = _SimHcPostFn.apply(x, residual_flat, post, comb, module_path)
        return y.view(dim_b, dim_s, dim_n, dim_d)
