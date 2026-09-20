# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only simulator shims for mm_gc model-specific fused ops.

Production kernels are the planned triton-ascend SLA2/MagiMoE kernels (planned
names, to be registered in the cost model together with
``torchtitan_npu/models/mm_gc/MODEL_CONTRACT.md``):

- ``triton_ascend_kernels.sla2_block_route_topk``: block pool + router proj +
  top-k block selection. Output count discriminates the stage: 1 output =
  stage 1 (soft mask, float32 ``[B, H, Nb_q, Nb_k]``), 2 outputs = stage 2
  (int8 ``sparse_map`` + int64 ``lut [B, H, Nb_q, K_eff]``).
- ``triton_ascend_kernels.sla2_sparse_attn``: sparse softmax branch on the
  selected key blocks. 4 inputs; the dtype of the last input (selection)
  discriminates the stage: float32 soft mask (stage 1) vs int64 lut (stage 2).
- ``triton_ascend_kernels.sla2_linear_attn``: global linear-attention branch
  with softmax feature map. Inputs ``[q, k, v, (, selection,) meta]`` —
  4 in ``compute_mode="full"``, 5 in the default ``compute_mode="partial"``.
- ``triton_ascend_kernels.mh_moe_route_topk``: fused Multi-Head MoE routing
  (per-head router bmm + score func + expert-biased top-k + L1 route_norm +
  route_scale). top_k is recovered from the output shape (effective value),
  never from config.

``compute_mode`` ("full" | "partial", default "partial") is threaded from the
SLA2 attention config (``sla2_compute_mode``, top-level ``MMGcModel.Config``
field) into the ``sla2_sparse_attn`` / ``sla2_linear_attn`` op records:
- sparse: ``full`` costs L×L attention; ``partial`` costs only the selected
  key blocks (``skv``).
- linear: ``full`` is the global linear attention over the full sequence;
  ``partial`` (M_not complement) covers only the non-selected key blocks
  (``l_eff = L - skv``), matching ``SparseLinearAttention._calc_linear_masked``.

The effective key lengths ``skv`` and ``l_eff`` are computed in shim-time
(``_sla_effective_key_lengths``) and appended to the op record's ``inputs``
as plain Python ints: sparse ``[q,k,v,selection,mode_val,stage,skv]``, linear
``[q,k,v,(,selection,)mode_val,l_eff]`` (mode_val=1 partial, 2 full). The cost
model reads them from ``op.inputs[-3]/-2/-1`` (sparse) and ``op.inputs[-2]/-1``
(linear) directly — no attrs, no tensor-value, no dtype inference.


The SLA2 alpha blend stays eager (production keeps it as small elementwise
ops). Follows kda_shim.py conventions: real op names recorded into the active
OpDispatchCapture with analytically-derived shapes; meta-safe (no data reads);
autograd Functions record ``*_grad`` in backward and return one gradient per
differentiable input; binding happens after model construction and
parallelization via ``apply_mm_gc_shims(model)`` with marker attributes.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.mm_gc.core import SparseLinearAttention
from torchtitan_npu.models.mm_gc.feed_forward import MultiHeadMoEGate
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

_SLA_CORE_SHIM_MARKER = "_simulator_mm_gc_sla_core_shim_installed"
_MH_MOE_GATE_SHIM_MARKER = "_simulator_mm_gc_mh_moe_gate_shim_installed"


def _record(
    raw_op_type: str,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    module_path: str,
    attrs: dict | None = None,
) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path,
            attrs=attrs,
        )


def _uncaptured_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def _uncaptured_empty(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=dtype, device=device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=dtype, device=device)


def _uncaptured_to(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return tensor.to(dtype)
    with capture.suspend_recording():
        return tensor.to(dtype)


def _current_module_path() -> str:
    capture = get_active_capture()
    if capture is not None and capture.module_path_tracker is not None:
        return capture.module_path_tracker.current_path()
    return ""


class _SimSla2BlockRouteTopk(torch.autograd.Function):
    @staticmethod
    def forward(  # noqa: ANN001
        ctx,
        q,
        k,
        proj_q_weight,
        proj_q_bias,
        proj_k_weight,
        proj_k_bias,
        blkq,
        blkk,
        topk_blocks,
        stage,
        module_path,
    ):
        B, H, L, _ = q.shape
        nb_q = (L + blkq - 1) // blkq
        nb_k = (L + blkk - 1) // blkk
        topk_eff = min(nb_k, topk_blocks)

        if stage == 1:
            sparse_map = _uncaptured_empty(
                (B, H, nb_q, nb_k), torch.float32, q.device
            )
            outputs = [sparse_map]
        else:
            sparse_map = _uncaptured_empty((B, H, nb_q, nb_k), torch.int8, q.device)
            lut = _uncaptured_empty((B, H, nb_q, topk_eff), torch.int64, q.device)
            outputs = [sparse_map, lut]

        _record(
            "triton_ascend_kernels.sla2_block_route_topk",
            [q, k, proj_q_weight, proj_q_bias, proj_k_weight, proj_k_bias],
            outputs,
            module_path,
        )

        ctx.save_for_backward(q, k, proj_q_weight, proj_k_weight)
        ctx.module_path = module_path
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):  # noqa: ANN001
        q, k, proj_q_weight, proj_k_weight = ctx.saved_tensors
        d_q = _uncaptured_empty_like(q)
        d_k = _uncaptured_empty_like(k)
        d_proj_q_w = _uncaptured_empty_like(proj_q_weight)
        d_proj_q_b = _uncaptured_empty_like(proj_q_weight[:, 0])
        d_proj_k_w = _uncaptured_empty_like(proj_k_weight)
        d_proj_k_b = _uncaptured_empty_like(proj_k_weight[:, 0])
        _record(
            "triton_ascend_kernels.sla2_block_route_topk_grad",
            [*grad_outputs],
            [d_q, d_k, d_proj_q_w, d_proj_q_b, d_proj_k_w, d_proj_k_b],
            ctx.module_path,
        )
        return (
            d_q,
            d_k,
            d_proj_q_w,
            d_proj_q_b,
            d_proj_k_w,
            d_proj_k_b,
            None,
            None,
            None,
            None,
            None,
        )


class _SimSla2SparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(  # noqa: ANN001
        ctx, q, k, v, selection, compute_mode, stage, skv, module_path
    ):
        output = _uncaptured_empty_like(q)
        mode_val = 1 if compute_mode == "partial" else 2
        _record(
            "triton_ascend_kernels.sla2_sparse_attn",
            [q, k, v, selection, mode_val, stage, skv],
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, selection)
        ctx.module_path = module_path
        ctx.compute_mode = compute_mode
        ctx.stage = stage
        ctx.skv = skv
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, selection = ctx.saved_tensors
        d_q = _uncaptured_empty_like(q)
        d_k = _uncaptured_empty_like(k)
        d_v = _uncaptured_empty_like(v)
        mode_val = 1 if ctx.compute_mode == "partial" else 2
        _record(
            "triton_ascend_kernels.sla2_sparse_attn_grad",
            [q, k, v, selection, grad_output, mode_val, ctx.stage, ctx.skv],
            [d_q, d_k, d_v],
            ctx.module_path,
        )
        return d_q, d_k, d_v, None, None, None, None, None


class _SimSla2LinearAttn(torch.autograd.Function):
    @staticmethod
    def forward(  # noqa: ANN001
        ctx, q, k, v, selection, compute_mode, l_eff, module_path
    ):
        output = _uncaptured_empty_like(q)
        mode_val = 1 if compute_mode == "partial" else 2
        inputs = [q, k, v]
        if compute_mode == "partial":
            inputs.append(selection)
        inputs.append(mode_val)
        inputs.append(l_eff)
        _record(
            "triton_ascend_kernels.sla2_linear_attn",
            inputs,
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, selection)
        ctx.module_path = module_path
        ctx.compute_mode = compute_mode
        ctx.l_eff = l_eff
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, selection = ctx.saved_tensors
        d_q = _uncaptured_empty_like(q)
        d_k = _uncaptured_empty_like(k)
        d_v = _uncaptured_empty_like(v)
        mode_val = 1 if ctx.compute_mode == "partial" else 2
        inputs = [q, k, v]
        if ctx.compute_mode == "partial":
            inputs.append(selection)
        inputs.append(grad_output)
        inputs.append(mode_val)
        inputs.append(ctx.l_eff)
        _record(
            "triton_ascend_kernels.sla2_linear_attn_grad",
            inputs,
            [d_q, d_k, d_v],
            ctx.module_path,
        )
        return d_q, d_k, d_v, None, None, None, None


class _SimMhMoeRouteTopk(torch.autograd.Function):
    @staticmethod
    def forward(  # noqa: ANN001
        ctx,
        x_heads,
        router,
        expert_bias,
        top_k,
        score_func,
        route_norm,
        route_scale,
        module_path,
    ):
        N, S = x_heads.shape[0], x_heads.shape[1]
        topk_probs = _uncaptured_empty((N, S, top_k), torch.float32, x_heads.device)
        topk_indices = _uncaptured_empty((N, S, top_k), torch.int64, x_heads.device)
        _record(
            "triton_ascend_kernels.mh_moe_route_topk",
            [x_heads, router, expert_bias],
            [topk_probs, topk_indices],
            module_path,
        )
        ctx.save_for_backward(x_heads, router)
        ctx.module_path = module_path
        return topk_probs, topk_indices

    @staticmethod
    def backward(ctx, grad_probs, grad_indices):  # noqa: ANN001
        x_heads, router = ctx.saved_tensors
        d_x = _uncaptured_empty_like(x_heads)
        d_router = _uncaptured_empty_like(router)
        _record(
            "triton_ascend_kernels.mh_moe_route_topk_grad",
            [x_heads, router, grad_probs],
            [d_x, d_router],
            ctx.module_path,
        )
        return d_x, d_router, None, None, None, None, None, None


def _sla_effective_key_lengths(
    L: int, stage: int, topk_blocks: int, blkk: int, compute_mode: str
) -> tuple[int, int]:
    """Return ``(skv_sparse, l_eff_linear)`` 用于稀疏分支与线性分支的计算量建模。

    - full 模式：两分支均覆盖全序列，skv = l_eff = L。
    - partial + stage 1（soft mask）：两分支均覆盖全序列（soft mask 稠密）。
    - partial + stage 2（hard top-k）：稀疏分支只算被选中的 key 块
      （skv = min(L, topk_blocks × BLKK)），线性分支只算其余块
      （l_eff = L - skv）。
    """
    if compute_mode == "full" or stage == 1:
        return L, L
    skv_sparse = min(L, topk_blocks * blkk)
    l_eff_linear = max(0, L - skv_sparse)
    return skv_sparse, l_eff_linear


def _sim_sla2_forward(module, q, k, v, return_sparsity=False):  # noqa: ANN001
    B, H, L, _ = q.shape
    nb_k = (L + module.BLKK - 1) // module.BLKK
    topk_blocks = min(nb_k, int(module.topk * nb_k))
    module_path = _current_module_path()
    compute_mode = getattr(module, "compute_mode", "partial")
    skv_sparse, l_eff_linear = _sla_effective_key_lengths(
        L, module.stage, topk_blocks, module.BLKK, compute_mode
    )

    route_outputs = _SimSla2BlockRouteTopk.apply(
        q,
        k,
        module.proj_q.weight,
        module.proj_q.bias,
        module.proj_k.weight,
        module.proj_k.bias,
        module.BLKQ,
        module.BLKK,
        topk_blocks,
        module.stage,
        module_path,
    )

    q_c = _uncaptured_to(q, module.dtype)
    k_c = _uncaptured_to(k, module.dtype)
    v_c = _uncaptured_to(v, module.dtype)

    if module.stage == 1:
        selection = route_outputs[0]
    else:
        selection = route_outputs[1]

    o_s = _SimSla2SparseAttn.apply(
        q_c,
        k_c,
        v_c,
        selection,
        compute_mode,
        module.stage,
        skv_sparse,
        module_path,
    )
    o_l = _SimSla2LinearAttn.apply(
        q_c,
        k_c,
        v_c,
        selection,
        compute_mode,
        l_eff_linear,
        module_path,
    )

    block_indices = torch.arange(L, device=q.device) // module.BLKQ
    alpha_per_position = module.alpha[block_indices].view(1, 1, L, 1)

    o = (alpha_per_position * o_s + (1 - alpha_per_position) * o_l).to(module.dtype)
    return o


def _sim_mh_moe_gate(module, x_heads):  # noqa: ANN001
    module_path = _current_module_path()
    router = module.router
    if isinstance(router, DTensor):
        router = router.to_local()
    expert_bias = module.expert_bias.view(-1, module.experts_per_head)
    expert_bias = expert_bias[
        module.local_head_start : module.local_head_start + module.local_num_heads
    ].reshape(-1)
    return _SimMhMoeRouteTopk.apply(
        x_heads,
        router,
        expert_bias,
        module.top_k,
        module.score_func,
        module.route_norm,
        module.route_scale,
        module_path,
    )


def apply_mm_gc_shims(model) -> None:
    """Bind mm_gc shape-only shims while preserving module hooks."""
    for module in model.modules():
        if isinstance(module, SparseLinearAttention) and not getattr(
            module, _SLA_CORE_SHIM_MARKER, False
        ):
            module.forward = MethodType(_sim_sla2_forward, module)
            setattr(module, _SLA_CORE_SHIM_MARKER, True)
        elif isinstance(module, MultiHeadMoEGate) and not getattr(
            module, _MH_MOE_GATE_SHIM_MARKER, False
        ):
            module.forward = MethodType(_sim_mh_moe_gate, module)
            setattr(module, _MH_MOE_GATE_SHIM_MARKER, True)
