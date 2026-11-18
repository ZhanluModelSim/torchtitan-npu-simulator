# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatcher-visible Meta contract for system3's retained fusion boundaries.

The operators declare the reference ``swiglu(x, gate, clamp)`` activation, the
GDN2 chunked delta-rule scan, and the ISA top-K indexer selection plus
selected-key sparse attention as single capture boundaries, not executable
NPU kernels; real-device calls fail explicitly.
"""

import math

import torch

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)

# Only the final feature axis is defined at this boundary; rank does not
# identify batch or token axes (routed experts see rank-major token rows).
_SWIGLU_INPUTS = (
    ("x", TensorMeaning("x", ("feature",), unknown_leading_axes=True)),
    ("gate", TensorMeaning("gate", ("feature",), unknown_leading_axes=True)),
)
_SWIGLU_OUTPUT = TensorMeaning("activated_output", ("feature",), unknown_leading_axes=True)

_QKV_AXES = ("batch", "sequence", "head", "key_head_dim")
_V_AXES = ("batch", "sequence", "head", "value_head_dim")
_STATE_AXES = ("batch", "sequence", "head", "state_dim")
_GAIN_AXES = ("batch", "sequence", "head")
_MASK_AXES = ("batch", "sequence")
_GDN2_INPUTS = (
    ("q", TensorMeaning("query", _QKV_AXES)),
    ("k", TensorMeaning("key", _QKV_AXES)),
    ("v", TensorMeaning("value", _V_AXES)),
    ("alpha", TensorMeaning("read_gate", _QKV_AXES)),
    ("beta_erase", TensorMeaning("erase_gate", _STATE_AXES)),
    ("beta_write", TensorMeaning("write_gate", _STATE_AXES)),
    ("da_gain", TensorMeaning("read_gain", _GAIN_AXES)),
)
_GDN2_MASK = ("mask", TensorMeaning("padding_mask", _MASK_AXES, optional=True))
_GDN2_OUTPUT = TensorMeaning("attention_output", _V_AXES)


@torch.library.custom_op("system3_sim::swiglu_clamp", mutates_args=())
def swiglu_clamp(x: torch.Tensor, gate: torch.Tensor, clamp_val: float | None) -> torch.Tensor:
    raise RuntimeError("system3 virtual swiglu_clamp is available only for Meta simulation")


@swiglu_clamp.register_fake
def _swiglu_clamp_meta(x, gate, clamp_val):
    return torch.empty_like(x)


@torch.library.custom_op("system3_sim::swiglu_clamp_backward", mutates_args=())
def swiglu_clamp_backward(
    x: torch.Tensor, gate: torch.Tensor, dy: torch.Tensor, clamp_val: float | None
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual swiglu_clamp backward is available only for Meta simulation")


@swiglu_clamp_backward.register_fake
def _swiglu_clamp_backward_meta(x, gate, dy, clamp_val):
    return [torch.empty_like(x), torch.empty_like(gate)]


def _swiglu_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:2])
    ctx.clamp_val = inputs[2]


def _swiglu_backward(ctx, dy):
    return (*swiglu_clamp_backward(*ctx.saved_tensors, dy, ctx.clamp_val), None)


swiglu_clamp.register_autograd(_swiglu_backward, setup_context=_swiglu_setup)


@torch.library.custom_op("system3_sim::gdn2_chunked", mutates_args=())
def gdn2_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta_erase: torch.Tensor,
    beta_write: torch.Tensor,
    da_gain: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    raise RuntimeError("system3 virtual gdn2_chunked is available only for Meta simulation")


@gdn2_chunked.register_fake
def _gdn2_chunked_meta(q, k, v, alpha, beta_erase, beta_write, da_gain, mask):
    return torch.empty_like(v)


@torch.library.custom_op("system3_sim::gdn2_chunked_backward", mutates_args=())
def gdn2_chunked_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta_erase: torch.Tensor,
    beta_write: torch.Tensor,
    da_gain: torch.Tensor,
    dy: torch.Tensor,
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual gdn2_chunked backward is available only for Meta simulation")


@gdn2_chunked_backward.register_fake
def _gdn2_chunked_backward_meta(q, k, v, alpha, beta_erase, beta_write, da_gain, dy):
    return [torch.empty_like(t) for t in (q, k, v, alpha, beta_erase, beta_write, da_gain)]


def _gdn2_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:7])


def _gdn2_backward(ctx, dy):
    return (*gdn2_chunked_backward(*ctx.saved_tensors, dy), None)


gdn2_chunked.register_autograd(_gdn2_backward, setup_context=_gdn2_setup)


def _register_fusion_semantics(
    forward: torch.library.CustomOpDef,
    backward: torch.library.CustomOpDef,
) -> None:
    register_tensor_semantics(str(forward._opoverload), OperatorTensorSemantics(_SWIGLU_INPUTS, (_SWIGLU_OUTPUT,)))
    output_gradient = TensorMeaning("output_gradient", _SWIGLU_OUTPUT.axes, _SWIGLU_OUTPUT.unknown_leading_axes)
    gradients = tuple(
        TensorMeaning(f"{meaning.role}_gradient", meaning.axes, meaning.unknown_leading_axes)
        for _, meaning in _SWIGLU_INPUTS
    )
    register_tensor_semantics(
        str(backward._opoverload),
        OperatorTensorSemantics((*_SWIGLU_INPUTS, ("dy", output_gradient)), gradients),
    )


_register_fusion_semantics(swiglu_clamp, swiglu_clamp_backward)


def _register_gdn2_semantics() -> None:
    inputs = (*_GDN2_INPUTS, _GDN2_MASK)
    register_tensor_semantics(str(gdn2_chunked._opoverload), OperatorTensorSemantics(inputs, (_GDN2_OUTPUT,)))
    output_gradient = TensorMeaning("output_gradient", _GDN2_OUTPUT.axes, _GDN2_OUTPUT.unknown_leading_axes)
    gradients = tuple(
        TensorMeaning(f"{meaning.role}_gradient", meaning.axes, meaning.unknown_leading_axes)
        for _, meaning in _GDN2_INPUTS
    )
    register_tensor_semantics(
        str(gdn2_chunked_backward._opoverload),
        OperatorTensorSemantics((*_GDN2_INPUTS, ("dy", output_gradient)), gradients),
    )


_register_gdn2_semantics()


# ---------------------------------------------------------------------------
# ISA: stride-compressed indexer selection + selected-key sparse attention.
# The projections and RMSNorms stay visible; the causal top-K selection over
# the compressed grid and the gather/score/softmax/combine core are single
# capture boundaries.  The attention saves an LSE for the gradient like the
# selected-key attention precedent, and the indexer gradient flows through
# the top-K scores back into the projections.
# ---------------------------------------------------------------------------
_IDX_Q_AXES = ("batch", "sequence", "index_width")
_IDX_K_AXES = ("batch", "compressed_sequence", "index_width")
_IDX_SEL_AXES = ("batch", "sequence", "selected_key")
_IDX_Q = TensorMeaning("indexer_query", _IDX_Q_AXES)
_IDX_K = TensorMeaning("indexer_key", _IDX_K_AXES)
_IDX_MASK = ("mask", TensorMeaning("padding_mask", ("batch", "sequence"), optional=True))
_IDX_INDICES = TensorMeaning("topk_indices", _IDX_SEL_AXES)
_IDX_SCORES = TensorMeaning("topk_scores", _IDX_SEL_AXES)

_ISA_Q_AXES = ("batch", "sequence", "head", "qk_head_dim")
_ISA_K_AXES = ("batch", "compressed_sequence", "head", "qk_head_dim")
_ISA_V_AXES = ("batch", "compressed_sequence", "head", "value_head_dim")
_ISA_Q = TensorMeaning("query", _ISA_Q_AXES)
_ISA_K = TensorMeaning("key", _ISA_K_AXES)
_ISA_V = TensorMeaning("value", _ISA_V_AXES)
_ISA_MASK = ("mask", TensorMeaning("padding_mask", ("batch", "sequence"), optional=True))
_ISA_INDICES = TensorMeaning("selected_indices", _IDX_SEL_AXES)
_ISA_BIAS = TensorMeaning("indexer_scores", _IDX_SEL_AXES)
_ISA_OUT = TensorMeaning("attention_output", ("batch", "sequence", "head", "value_head_dim"))
_ISA_LSE = TensorMeaning("logsumexp", ("batch", "head", "sequence"))

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _isa_indexer_geometry(idx_q, idx_k_c):
    if idx_q.ndim != 3 or idx_k_c.ndim != 3 or min(*idx_q.shape, *idx_k_c.shape) <= 0:
        raise ValueError("ISA indexer requires positive B/S and index-width geometry")
    b, s, width = idx_q.shape
    if idx_k_c.shape[0] != b or idx_k_c.shape[2] != width or idx_k_c.shape[1] > s:
        raise ValueError("ISA indexer key grid must share batch and index width")
    if idx_q.dtype not in _FLOAT_DTYPES or idx_k_c.dtype != idx_q.dtype:
        raise ValueError("ISA indexer requires FP16, BF16 or FP32 query/key")
    return b, s, width, idx_k_c.shape[1]


@torch.library.custom_op("system3_sim::isa_indexer", mutates_args=())
def isa_indexer(
    idx_q: torch.Tensor,
    idx_k_c: torch.Tensor,
    mask: torch.Tensor | None,
    topk: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise RuntimeError("system3 virtual isa_indexer is available only for Meta simulation")


@isa_indexer.register_fake
def _isa_indexer_meta(idx_q, idx_k_c, mask, topk, compress_ratio):
    b, s, _width, cl = _isa_indexer_geometry(idx_q, idx_k_c)
    if topk < 1 or compress_ratio < 1:
        raise ValueError("ISA indexer requires positive top-k and compression ratio")
    if mask is not None and mask.shape != (b, s):
        raise ValueError("ISA indexer padding mask must cover the batch and sequence")
    if len({t.device for t in (idx_q, idx_k_c, *(() if mask is None else (mask,)))}) != 1:
        raise ValueError("ISA indexer operands must share placement")
    actual = min(topk, cl)
    indices = torch.empty((b, s, actual), dtype=torch.int64, device=idx_q.device)
    scores = torch.empty((b, s, actual), dtype=idx_q.dtype, device=idx_q.device)
    return indices, scores


@torch.library.custom_op("system3_sim::isa_indexer_grad", mutates_args=())
def isa_indexer_grad(
    idx_q: torch.Tensor, idx_k_c: torch.Tensor, indices: torch.Tensor, d_scores: torch.Tensor
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual isa_indexer backward is available only for Meta simulation")


@isa_indexer_grad.register_fake
def _isa_indexer_grad_meta(idx_q, idx_k_c, indices, d_scores):
    b, s, _width, cl = _isa_indexer_geometry(idx_q, idx_k_c)
    if indices.dtype != torch.int64 or indices.ndim != 3 or indices.shape[:2] != (b, s):
        raise ValueError("ISA indexer gradient requires B/S int64 selections")
    if indices.shape[-1] < 1 or indices.shape[-1] > cl:
        raise ValueError("ISA indexer selections must stay within the compressed grid")
    if d_scores.shape != indices.shape or d_scores.dtype != idx_q.dtype:
        raise ValueError("ISA indexer score gradient must match the selection geometry")
    if len({t.device for t in (idx_q, idx_k_c, indices, d_scores)}) != 1:
        raise ValueError("ISA indexer gradient operands must share placement")
    return [torch.empty_like(idx_q), torch.empty_like(idx_k_c)]


def _isa_indexer_setup(ctx, inputs, output):
    idx_q, idx_k_c, _mask, _topk, _compress_ratio = inputs
    indices, scores = output
    ctx.save_for_backward(idx_q, idx_k_c, indices, scores)
    ctx.mark_non_differentiable(indices)


def _isa_indexer_backward(ctx, grad_indices, grad_scores):
    idx_q, idx_k_c, indices, _scores = ctx.saved_tensors
    dq, dk = isa_indexer_grad(idx_q, idx_k_c, indices, grad_scores)
    return (dq, dk, None, None, None)


isa_indexer.register_autograd(_isa_indexer_backward, setup_context=_isa_indexer_setup)


def _isa_attention_geometry(q, k, v, indices, bias):
    if q.ndim != 4 or min(*q.shape) <= 0 or q.dtype not in _FLOAT_DTYPES:
        raise ValueError("ISA attention requires positive floating-point B/S/H/D query")
    b, s, heads, width = q.shape
    if k.ndim != 4 or k.shape[0] != b or k.shape[2:] != (heads, width) or min(k.shape) <= 0:
        raise ValueError("ISA attention key must match the query's batch, heads and width")
    if k.dtype != q.dtype:
        raise ValueError("ISA attention key must share the query precision")
    if v.ndim != 4 or v.shape[:3] != k.shape[:3] or v.shape[-1] <= 0 or v.dtype != q.dtype:
        raise ValueError("ISA attention value must match the key's batch/grid/heads")
    if indices.dtype != torch.int64 or indices.ndim != 3 or indices.shape[:2] != (b, s):
        raise ValueError("ISA attention indices must describe one selection row per query")
    if min(indices.shape[-1], k.shape[1]) < 1 or indices.shape[-1] > k.shape[1]:
        raise ValueError("ISA attention selections must stay within the compressed grid")
    if bias.shape != indices.shape or bias.dtype != q.dtype:
        raise ValueError("ISA indexer bias must match the selection geometry and precision")
    return b, s, heads, width, k.shape[1]


@torch.library.custom_op("system3_sim::isa_sparse_attention", mutates_args=())
def isa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    clamp_val: float,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise RuntimeError("system3 virtual isa_sparse_attention is available only for Meta simulation")


@isa_sparse_attention.register_fake
def _isa_sparse_attention_meta(q, k, v, indices, bias, mask, scale, clamp_val, compress_ratio):
    b, s, _heads, _width, _cl = _isa_attention_geometry(q, k, v, indices, bias)
    if not (scale > 0) or not math.isfinite(scale):
        raise ValueError("ISA attention requires a finite positive scale")
    if not (clamp_val > 0) or not math.isfinite(clamp_val):
        raise ValueError("ISA attention requires a finite positive softmax clamp")
    if compress_ratio < 1:
        raise ValueError("ISA attention requires a positive compression ratio")
    if mask is not None and mask.shape != (b, s):
        raise ValueError("ISA attention padding mask must cover the batch and sequence")
    devices = {t.device for t in (q, k, v, indices, bias, *(() if mask is None else (mask,)))}
    if len(devices) != 1:
        raise ValueError("ISA attention operands must share placement")
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[2], s), dtype=torch.float32, device=q.device)
    return out, lse


@torch.library.custom_op("system3_sim::isa_sparse_attention_grad", mutates_args=())
def isa_sparse_attention_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dy: torch.Tensor,
    scale: float,
    clamp_val: float,
    compress_ratio: int,
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual isa_sparse_attention backward is available only for Meta simulation")


@isa_sparse_attention_grad.register_fake
def _isa_sparse_attention_grad_meta(q, k, v, indices, bias, out, lse, dy, scale, clamp_val, compress_ratio):
    _isa_attention_geometry(q, k, v, indices, bias)
    if out.shape != q.shape or out.dtype != q.dtype or dy.shape != q.shape or dy.dtype != q.dtype:
        raise ValueError("ISA attention saved output and gradient must match the query")
    if lse.dtype != torch.float32 or lse.shape != (q.shape[0], q.shape[2], q.shape[1]):
        raise ValueError("ISA attention saved LSE must be B/H/S float32")
    devices = {t.device for t in (q, k, v, indices, bias, out, lse, dy)}
    if len(devices) != 1:
        raise ValueError("ISA attention gradient operands must share placement")
    return [torch.empty_like(t) for t in (q, k, v, bias)]


def _isa_attention_setup(ctx, inputs, output):
    q, k, v, indices, bias, _mask, scale, clamp_val, compress_ratio = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, indices, bias, out, lse)
    ctx.scale, ctx.clamp_val, ctx.compress_ratio = scale, clamp_val, compress_ratio
    ctx.mark_non_differentiable(lse)


def _isa_attention_backward(ctx, d_out, _unused_lse_gradient):
    q, k, v, indices, bias, out, lse = ctx.saved_tensors
    dq, dk, dv, dbias = isa_sparse_attention_grad(
        q, k, v, indices, bias, out, lse, d_out, ctx.scale, ctx.clamp_val, ctx.compress_ratio
    )
    return (dq, dk, dv, None, dbias, None, None, None, None)


isa_sparse_attention.register_autograd(_isa_attention_backward, setup_context=_isa_attention_setup)


def _register_isa_semantics() -> None:
    indexer_inputs = (("idx_q", _IDX_Q), ("idx_k_c", _IDX_K), _IDX_MASK)
    register_tensor_semantics(
        str(isa_indexer._opoverload),
        OperatorTensorSemantics(indexer_inputs, (_IDX_INDICES, _IDX_SCORES)),
    )
    register_tensor_semantics(
        str(isa_indexer_grad._opoverload),
        OperatorTensorSemantics(
            (
                ("idx_q", _IDX_Q),
                ("idx_k_c", _IDX_K),
                ("indices", _IDX_INDICES),
                ("d_scores", TensorMeaning("topk_scores_gradient", _IDX_SCORES.axes)),
            ),
            (
                TensorMeaning("indexer_query_gradient", _IDX_Q.axes),
                TensorMeaning("indexer_key_gradient", _IDX_K.axes),
            ),
        ),
    )
    attention_inputs = (
        ("q", _ISA_Q),
        ("k", _ISA_K),
        ("v", _ISA_V),
        ("indices", _ISA_INDICES),
        ("bias", _ISA_BIAS),
        _ISA_MASK,
    )
    register_tensor_semantics(
        str(isa_sparse_attention._opoverload),
        OperatorTensorSemantics(attention_inputs, (_ISA_OUT, _ISA_LSE)),
    )
    register_tensor_semantics(
        str(isa_sparse_attention_grad._opoverload),
        OperatorTensorSemantics(
            (
                ("q", _ISA_Q),
                ("k", _ISA_K),
                ("v", _ISA_V),
                ("indices", _ISA_INDICES),
                ("bias", _ISA_BIAS),
                ("out", _ISA_OUT),
                ("lse", _ISA_LSE),
                ("dy", TensorMeaning("output_gradient", _ISA_OUT.axes)),
            ),
            (
                TensorMeaning("query_gradient", _ISA_Q.axes),
                TensorMeaning("key_gradient", _ISA_K.axes),
                TensorMeaning("value_gradient", _ISA_V.axes),
                TensorMeaning("indexer_scores_gradient", _ISA_BIAS.axes),
            ),
        ),
    )


_register_isa_semantics()


def install_system3_virtual_fusions(scope):
    """Replace the model-level fusion call sites with the virtual boundaries."""
    import torchtitan_npu.models.system3.model as system3_model
    from torchtitan_npu.models.system3.model import GDN2Attention, IndexedSparseAttention, ISAIndexer

    def activation(x, gate, clamp_val=None):
        return swiglu_clamp(x, gate, clamp_val)

    def scan(module, q, k, v, alpha, beta_erase, beta_write, da_gain, mask):
        return gdn2_chunked(q, k, v, alpha, beta_erase, beta_write, da_gain, mask)

    def indexer_topk(module, idx_q, idx_k_c, attention_mask=None):
        return isa_indexer(idx_q, idx_k_c, attention_mask, module.topk, module.compress_ratio)

    def selected_attention(module, q_full, k_comp, v_comp, topk_indices, topk_scores, attention_mask=None):
        out, _lse = isa_sparse_attention(
            q_full,
            k_comp,
            v_comp,
            topk_indices,
            topk_scores,
            attention_mask,
            module.scale,
            module.softmax_clamp,
            module.compress_ratio,
        )
        return out

    scope.replace(system3_model, "swiglu", activation)
    scope.replace(GDN2Attention, "_delta_rule_scan", scan)
    scope.replace(ISAIndexer, "_select_topk", indexer_topk)
    scope.replace(IndexedSparseAttention, "_sparse_attention", selected_attention)
