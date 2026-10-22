# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim mirroring MindSpeed's triton GDN kernel sequence.

The production path (``use_triton_gdn=True``) binds
``mindspeed_llm.tasks.models.transformer.chunk_gated_delta_rule`` which
launches one triton-ascend kernel per stage instead of the eager torch
fallback. This shim records each kernel in the exact production order
(mirrors kda_shim.py pattern) AND executes the wrapper-level eager ops the
reference triton wrappers perform around every kernel, because those are
real captured device work on NPU (allocations, layout contiguous copies,
fills, reductions, casts):

forward:  l2norm_fwd(q) l2norm_fwd(k) chunk_local_cumsum
          chunk_scaled_dot_kkt_fwd solve_tril recompute_w_u_fwd
          chunk_gated_delta_rule_fwd_h chunk_fwd_o
backward: recompute_w_u_fwd chunk_gated_delta_rule_fwd_h chunk_bwd_dv_local
          chunk_gated_delta_rule_bwd_dhu chunk_bwd_dqkwg
          prepare_wy_repr_bwd add_(dk2) add_(dg2) chunk_local_cumsum(reverse)
          l2norm_bwd(q) l2norm_bwd(k) + output dtype casts

Per-stage wrapper-level ops mirrored from ``mindspeed.ops.triton``:
- l2norm_fwd/bwd: ``empty_like(x)`` output + ``empty((T,), f32)`` rstd
  (bwd: ``empty_like`` dx only), driven through ``view(-1, D)`` 2D form
- chunk_local_cumsum: ``empty_like(g, dtype=...)`` output
- chunk_scaled_dot_kkt_fwd: ``beta/g.transpose(1,2).contiguous()`` +
  ``A = torch.empty(B,T,H,BT)`` (gk=None branch)
- solve_tril: ``Ai = torch.zeros_like(A, dtype=k.dtype)``
- recompute_w_u_fwd: ``g/beta.transpose(1,2).contiguous()`` +
  ``w/u = torch.empty_like(k/v)``
- chunk_gated_delta_rule_fwd_h: ``h = k.new_empty(B,NT,H,K,V).permute.contiguous()``,
  ``v_new = empty_like(u).permute.contiguous()``,
  ``k/w/u/g.permute.contiguous()`` layout copies in, ``h/v_new.permute.contiguous()``
  layout copies out (``output_final_state=False`` in training: no final_state)
- chunk_fwd_o: ``o = empty_like(v)``, ``g.transpose(1,2).contiguous()``,
  ``h.contiguous()``
- chunk_bwd_dv_local: ``g.transpose(1,2).contiguous()``, ``dv = empty_like(do)``
- chunk_gated_delta_rule_bwd_dhu: ``dh = q.new_empty(B,NT,H,K,V)``,
  ``dv2 = empty_like(dv)``, ``g.permute(0,2,1).contiguous()``
- chunk_bwd_dqkwg: ``dq/dk/dw = empty_like(q/k/w)``,
  ``dg = torch.empty(NK,*g.shape, f32)``, ``g.transpose(1,2).contiguous()``,
  post-kernel ``dg.sum(0)`` + ``dg.transpose(1,2).contiguous()``
- prepare_wy_repr_bwd: ``beta.transpose(1,2).contiguous()`` +
  ``dk/dv/db/dg = empty_like`` x4
- ChunkGatedDeltaRuleFunction: ``dk.add_(dk2)``, ``dg.add_(dg2)``,
  output casts ``dq/dk/dv/dg/db .to(...)``

Shape note: the g-family layout conversions are recorded with the module's
``[B,T,H]`` convention (the reference's in-wrapper transpose dance produces
equivalent op types/counts; exact intermediate strides differ between the
pinned reference and the server build).
"""

from __future__ import annotations

import torch

from torchtitan_npu.models.qwen3_next.gated_deltanet import GatedDeltaNet
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

_GDN_SHIM_MARKER = "_simulator_gdn_shim_installed"

_TRITON = "mindspeed.ops.triton"


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


def _current_module_path() -> str:
    capture = get_active_capture()
    if capture is not None and capture.module_path_tracker is not None:
        return capture.module_path_tracker.current_path()
    return ""


def _l2norm_fwd(x: torch.Tensor, module_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``l2norm_fwd``: 2D view, empty_like output, empty f32 rstd."""
    x2d = x.view(-1, x.shape[-1])
    y2d = torch.empty_like(x2d)
    rstd = torch.empty((x2d.shape[0],), dtype=torch.float32, device=x.device)
    _record(
        f"{_TRITON}.l2norm_fwd",
        [x],
        [y2d.view(x.shape), rstd.view(x.shape[:-1])],
        module_path,
    )
    return y2d.view(x.shape), rstd.view(x.shape[:-1])


def _l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, module_path: str) -> torch.Tensor:
    """Reference ``l2norm_bwd``: 2D views + f32 dx (the triton bwd kernel
    computes in f32; the wrapper's ``.to()`` downcast is real device work)."""
    y2d = y.view(-1, y.shape[-1])
    dy2d = dy.view(-1, dy.shape[-1])
    dx2d = torch.empty(y2d.shape, dtype=torch.float32, device=y.device)
    _record(f"{_TRITON}.l2norm_bwd", [y, rstd, dy], [dx2d.view(y.shape)], module_path)
    return dx2d.view(y.shape)


def _recompute_w_u(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, a: torch.Tensor,
                   g_cs: torch.Tensor, module_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``recompute_w_u_fwd`` wrapper: g/beta contiguous + empty_like w/u."""
    g_c = g_cs.transpose(1, 2).contiguous()
    beta_c = beta.transpose(1, 2).contiguous()
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    _record(f"{_TRITON}.recompute_w_u_fwd", [k, v, beta_c, a, g_c], [w, u], module_path)
    return w, u


def _fwd_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cs: torch.Tensor,
           nc: int, module_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``chunk_gated_delta_rule_fwd_h`` wrapper: new_empty h +
    empty_like v_new with permute/contiguous layout copies in and out."""
    b, t, h, kd = k.shape
    vd = u.shape[-1]
    h_state = k.new_empty(b, nc, h, kd, vd).permute(0, 2, 1, 3, 4).contiguous()
    v_new = torch.empty_like(u).permute(0, 2, 1, 3).contiguous()
    k_c = k.permute(0, 2, 1, 3).contiguous()
    w_c = w.permute(0, 2, 1, 3).contiguous()
    u_c = u.permute(0, 2, 1, 3).contiguous()
    g_c = g_cs.permute(0, 2, 1).contiguous()
    _record(
        f"{_TRITON}.chunk_gated_delta_rule_fwd_h",
        [k_c, w_c, u_c, g_c],
        [h_state, v_new],
        module_path,
    )
    h_state = h_state.permute(0, 2, 1, 3, 4).contiguous()
    v_new = v_new.permute(0, 2, 1, 3).contiguous()
    return h_state, v_new


class _SimTritonChunkGatedDeltaRuleFn(torch.autograd.Function):
    """Shape-only autograd bridge for the MindSpeed triton GDN kernel sequence."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, q, k, v, g, beta, module_path, chunk_size):
        b, t, h, kd = q.shape
        vd = v.shape[-1]
        nc = (t + chunk_size - 1) // chunk_size
        dev = q.device

        q_norm, q_rstd = _l2norm_fwd(q, module_path)
        k_norm, k_rstd = _l2norm_fwd(k, module_path)

        g_cs = torch.empty_like(g)
        _record(f"{_TRITON}.chunk_local_cumsum", [g], [g_cs], module_path)

        # kkt (gk=None branch): beta/g contiguous copies + empty A (f32)
        beta_c = beta.transpose(1, 2).contiguous()
        g_c = g_cs.transpose(1, 2).contiguous()
        a = torch.empty(b, t, h, chunk_size, device=dev, dtype=torch.float32)
        _record(f"{_TRITON}.chunk_scaled_dot_kkt_fwd", [k_norm, g_c, beta_c], [a], module_path)

        # solve_tril: zeros_like with output_dtype=k.dtype
        ai = torch.zeros_like(a, dtype=k.dtype)
        _record(f"{_TRITON}.solve_tril", [a], [ai], module_path)
        a = ai

        w, u = _recompute_w_u(k_norm, v, beta, a, g_cs, module_path)
        h_state, v_new = _fwd_h(k_norm, w, u, g_cs, nc, module_path)

        # fwd_o: f32 output (triton kernel computes in f32) + g
        # transpose-contiguous + h contiguous; the wrapper's o.to(q.dtype)
        # downcast below is a real cast
        o = torch.empty(v_new.shape, dtype=torch.float32, device=dev)
        g_c = g_cs.transpose(1, 2).contiguous()
        h_c = h_state.contiguous()
        _record(f"{_TRITON}.chunk_fwd_o", [q_norm, k_norm, v_new, h_c, g_c], [o], module_path)

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, a, g_cs, w, u, h_state, v_new)
        ctx.module_path = module_path
        ctx.chunk_size = chunk_size
        return o.to(q.dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, do):
        q, q_rstd, k, k_rstd, v, g, beta, a, g_cs, w, u, h_state, v_new = ctx.saved_tensors
        module_path = ctx.module_path
        chunk_size = ctx.chunk_size
        b, t, h, kd = q.shape
        vd = v.shape[-1]
        nc = (t + chunk_size - 1) // chunk_size
        dev = q.device

        w2, u2 = _recompute_w_u(k, v, beta, a, g_cs, module_path)
        h2, v_new2 = _fwd_h(k, w2, u2, g_cs, nc, module_path)

        # bwd_dv_local: g contiguous copy + empty_like dv
        g_c = g_cs.transpose(1, 2).contiguous()
        dv = torch.empty_like(do)
        _record(f"{_TRITON}.chunk_bwd_dv_local", [q, k, g_c, do], [dv], module_path)

        # bwd_dhu: dh new_empty + dv2 empty_like + g contiguous copy
        dh = q.new_empty(b, nc, h, kd, vd)
        dv2 = torch.empty_like(dv)
        g_c = g_cs.permute(0, 2, 1).contiguous()
        _record(
            f"{_TRITON}.chunk_gated_delta_rule_bwd_dhu",
            [q, k, w2, g_c, do, dv],
            [dh, dv2],
            module_path,
        )
        dv = dv2

        # bwd_dqkwg: empty_like dq/dk/dw + empty f32 dg (post-g-conversion
        # shape [1,B,H,T], NK=1) + g contiguous copy, post-kernel sum(0) +
        # transpose-contiguous landing back on [B,T,H]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        g_c = g_cs.transpose(1, 2).contiguous()
        dg = torch.empty(1, b, h, t, dtype=torch.float32, device=dev)
        dw = torch.empty_like(w2)
        _record(
            f"{_TRITON}.chunk_bwd_dqkwg",
            [q, k, v_new2, w2, h2, dv, do, dh],
            [dq, dk, dw, dg],
            module_path,
        )
        dg = dg.sum(0)
        dg = dg.transpose(1, 2).contiguous()

        # prepare_wy_repr_bwd: beta contiguous copy + empty_like dk/db/dg,
        # f32 dv (kernel computes in f32; dv.to(v) downcast is real)
        beta_c = beta.transpose(1, 2).contiguous()
        dk2 = torch.empty_like(k)
        dv3 = torch.empty(v.shape, dtype=torch.float32, device=dev)
        db = torch.empty_like(beta)
        dg2 = torch.empty_like(dg)
        _record(
            f"{_TRITON}.prepare_wy_repr_bwd",
            [k, v, beta_c, g_cs, a, dw, dv],
            [dk2, dv3, db, dg2],
            module_path,
        )
        dv = dv3
        # eager in the reference file: dk.add_(dk2); dg.add_(dg2)
        dk.add_(dk2)
        dg.add_(dg2)

        dg_cs = torch.empty_like(dg)
        _record(f"{_TRITON}.chunk_local_cumsum", [dg], [dg_cs], module_path)

        dq = _l2norm_bwd(q, q_rstd, dq, module_path)
        dk = _l2norm_bwd(k, k_rstd, dk, module_path)
        # eager output casts in the reference file
        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)
        dg = dg_cs.to(g.dtype)
        db = db.to(beta.dtype)
        return dq, dk, dv, dg, db, None, None  # module_path, chunk_size


def sim_chunk_gated_delta_rule(module: GatedDeltaNet, q, k, v, g, beta):
    """Record the MindSpeed triton GDN kernel sequence (production path)."""
    module_path = _current_module_path()
    return _SimTritonChunkGatedDeltaRuleFn.apply(
        q,
        k,
        v,
        g,
        beta,
        module_path,
        module.gdn_chunk_size,
    )


def apply_qwen3_next_shims(model) -> None:
    """Bind the GDN triton-kernel-sequence shim while preserving module hooks."""
    from types import MethodType

    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            if not getattr(module, _GDN_SHIM_MARKER, False):
                module._chunk_gated_delta_rule = MethodType(
                    sim_chunk_gated_delta_rule, module
                )
                setattr(module, _GDN_SHIM_MARKER, True)
