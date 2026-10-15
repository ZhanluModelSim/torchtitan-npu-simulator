# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim mirroring MindSpeed's triton GDN kernel sequence.

The production path (``use_triton_gdn=True``) binds
``mindspeed_llm.tasks.models.transformer.chunk_gated_delta_rule`` which
launches one triton-ascend kernel per stage instead of the eager torch
fallback. This shim records each kernel in the exact production order
with analytically-derived shapes (mirrors kda_shim.py pattern):

forward:  l2norm_fwd(q) l2norm_fwd(k) chunk_local_cumsum
          chunk_scaled_dot_kkt_fwd solve_tril recompute_w_u_fwd
          chunk_gated_delta_rule_fwd_h chunk_fwd_o
backward: recompute_w_u_fwd chunk_gated_delta_rule_fwd_h chunk_bwd_dv_local
          chunk_gated_delta_rule_bwd_dhu chunk_bwd_dqkwg
          prepare_wy_repr_bwd add_(dk2) add_(dg2) chunk_local_cumsum(reverse)
          l2norm_bwd(q) l2norm_bwd(k) + output dtype casts
"""

from __future__ import annotations

import torch

from torchtitan_npu.models.qwen3_next.gated_deltanet import GatedDeltaNet
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

_GDN_SHIM_MARKER = "_simulator_gdn_shim_installed"

_TRITON = "mindspeed.ops.triton"


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


def _uncaptured_empty(*shape, dtype=torch.float32, device=None) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty(*shape, dtype=dtype, device=device)
    with capture.suspend_recording():
        return torch.empty(*shape, dtype=dtype, device=device)


def _uncaptured_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def _current_module_path() -> str:
    capture = get_active_capture()
    if capture is not None and capture.module_path_tracker is not None:
        return capture.module_path_tracker.current_path()
    return ""


class _SimTritonChunkGatedDeltaRuleFn(torch.autograd.Function):
    """Shape-only autograd bridge for the MindSpeed triton GDN kernel sequence."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, q, k, v, g, beta, module_path, chunk_size):
        b, t, h, kd = q.shape
        vd = v.shape[-1]
        nc = (t + chunk_size - 1) // chunk_size
        dev = q.device

        q_norm = _uncaptured_empty_like(q)
        q_rstd = _uncaptured_empty(b, t, h, 1, device=dev)
        _record(f"{_TRITON}.l2norm_fwd", [q], [q_norm, q_rstd], module_path)
        k_norm = _uncaptured_empty_like(k)
        k_rstd = _uncaptured_empty(b, t, h, 1, device=dev)
        _record(f"{_TRITON}.l2norm_fwd", [k], [k_norm, k_rstd], module_path)

        g_cs = _uncaptured_empty(b, t, h, device=dev)
        _record(f"{_TRITON}.chunk_local_cumsum", [g], [g_cs], module_path)
        a = _uncaptured_empty(b, nc, h, chunk_size, chunk_size, device=dev)
        _record(f"{_TRITON}.chunk_scaled_dot_kkt_fwd", [k_norm, g_cs, beta], [a], module_path)
        _record(f"{_TRITON}.solve_tril", [a], [a], module_path)
        w = _uncaptured_empty(b, t, h, kd, device=dev)
        u = _uncaptured_empty(b, t, h, vd, device=dev)
        _record(f"{_TRITON}.recompute_w_u_fwd", [k_norm, v, beta, a, g_cs], [w, u], module_path)
        h_state = _uncaptured_empty(b, nc, h, kd, vd, device=dev)
        v_new = _uncaptured_empty(b, t, h, vd, device=dev)
        final_state = _uncaptured_empty(b, h, kd, vd, device=dev)
        _record(
            f"{_TRITON}.chunk_gated_delta_rule_fwd_h",
            [k_norm, w, u, g_cs],
            [h_state, v_new, final_state],
            module_path,
        )
        o = _uncaptured_empty_like(v)
        _record(f"{_TRITON}.chunk_fwd_o", [q_norm, k_norm, v_new, h_state, g_cs], [o], module_path)

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, a, g_cs, w, u, h_state, v_new)
        ctx.module_path = module_path
        ctx.chunk_size = chunk_size
        return o

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

        w2 = _uncaptured_empty(b, t, h, kd, device=dev)
        u2 = _uncaptured_empty(b, t, h, vd, device=dev)
        _record(f"{_TRITON}.recompute_w_u_fwd", [k, v, beta, a, g_cs], [w2, u2], module_path)
        h2 = _uncaptured_empty(b, nc, h, kd, vd, device=dev)
        v_new2 = _uncaptured_empty(b, t, h, vd, device=dev)
        fs2 = _uncaptured_empty(b, h, kd, vd, device=dev)
        _record(
            f"{_TRITON}.chunk_gated_delta_rule_fwd_h",
            [k, w2, u2, g_cs],
            [h2, v_new2, fs2],
            module_path,
        )
        dv = _uncaptured_empty(b, t, h, vd, device=dev)
        _record(f"{_TRITON}.chunk_bwd_dv_local", [q, k, g_cs, do], [dv], module_path)
        dh = _uncaptured_empty(b, nc, h, kd, vd, device=dev)
        dh0 = _uncaptured_empty(b, h, kd, vd, device=dev)
        _record(
            f"{_TRITON}.chunk_gated_delta_rule_bwd_dhu",
            [q, k, w2, g_cs, do, dv],
            [dh, dh0, dv],
            module_path,
        )
        dq = _uncaptured_empty(b, t, h, kd, device=dev)
        dk = _uncaptured_empty(b, t, h, kd, device=dev)
        dw = _uncaptured_empty(b, t, h, kd, device=dev)
        dg = _uncaptured_empty(b, t, h, device=dev)
        _record(
            f"{_TRITON}.chunk_bwd_dqkwg",
            [q, k, v_new2, w2, g_cs, h2, dv, do, dh],
            [dq, dk, dw, dg],
            module_path,
        )
        dk2 = _uncaptured_empty(b, t, h, kd, device=dev)
        dv2 = _uncaptured_empty(b, t, h, vd, device=dev)
        db = _uncaptured_empty(b, t, h, device=dev)
        dg2 = _uncaptured_empty(b, t, h, device=dev)
        _record(
            f"{_TRITON}.prepare_wy_repr_bwd",
            [k, v, beta, g_cs, a, dw, dv],
            [dk2, dv2, db, dg2],
            module_path,
        )
        # eager in the reference file: dk.add_(dk2); dg.add_(dg2)
        dk.add_(dk2)
        dg.add_(dg2)
        dg_cs = _uncaptured_empty(b, t, h, device=dev)
        _record(f"{_TRITON}.chunk_local_cumsum", [dg], [dg_cs], module_path)
        dq_raw, dk_raw = dq, dk
        dq = _uncaptured_empty(b, t, h, kd, device=dev)
        _record(f"{_TRITON}.l2norm_bwd", [q, q_rstd, dq_raw], [dq], module_path)
        dk = _uncaptured_empty(b, t, h, kd, device=dev)
        _record(f"{_TRITON}.l2norm_bwd", [k, k_rstd, dk_raw], [dk], module_path)
        # eager output casts in the reference file
        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv2.to(v.dtype)
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
