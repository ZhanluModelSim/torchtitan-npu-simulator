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
import os
import sys

import torch
import torch_npu

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
    idx_q: torch.Tensor,
    idx_k_c: torch.Tensor,
    indices: torch.Tensor,
    d_scores: torch.Tensor,
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual isa_indexer backward is available only for Meta simulation")


@isa_indexer_grad.register_fake
def _isa_indexer_grad_meta(idx_q, idx_k_c, indices, d_scores):
    b, s, _width, cl = _isa_indexer_geometry(idx_q, idx_k_c)
    if indices.ndim != 3 or indices.shape[:2] != (b, s):
        raise ValueError("ISA indexer gradient requires B/S selections")
    if indices.shape[-1] < 1 or indices.shape[-1] > cl:
        raise ValueError("ISA indexer selections must stay within the compressed grid")
    if d_scores.shape != indices.shape or d_scores.dtype != idx_q.dtype:
        raise ValueError("ISA indexer score gradient must match the selection score precision")
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
    if os.environ.get("SYSTEM3_DEBUG_ISA"):
        print(
            f"[isa] q={q.dtype}{tuple(q.shape)} k={k.dtype}{tuple(k.shape)} "
            f"v={v.dtype} bias={bias.dtype}",
            file=sys.stderr,
        )
    if q.ndim != 4 or min(*q.shape) <= 0 or q.dtype not in _FLOAT_DTYPES:
        raise ValueError("ISA attention requires positive floating-point B/S/H/D query")
    b, s, heads, width = q.shape
    if k.ndim != 4 or k.shape[0] != b or k.shape[2:] != (heads, width) or min(k.shape) <= 0:
        raise ValueError("ISA attention key must match the query's batch, heads and width")
    if k.dtype != q.dtype:
        raise ValueError(
            f"ISA attention key must share the query precision (q={q.dtype} {tuple(q.shape)}, "
            f"k={k.dtype} {tuple(k.shape)}, v={getattr(v, 'dtype', None)})"
        )
    if v.ndim != 4 or v.shape[:3] != k.shape[:3] or v.shape[-1] <= 0 or v.dtype != q.dtype:
        raise ValueError("ISA attention value must match the key's batch/grid/heads")
    if indices.dtype != torch.int64 or indices.ndim != 3 or indices.shape[:2] != (b, s):
        raise ValueError("ISA attention indices must describe one selection row per query")
    if min(indices.shape[-1], k.shape[1]) < 1 or indices.shape[-1] > k.shape[1]:
        raise ValueError("ISA attention selections must stay within the compressed grid")
    if bias.shape != indices.shape or bias.dtype not in (q.dtype, torch.float32):
        raise ValueError("ISA indexer bias must match the selection geometry in query or float32 precision")
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
    out = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=q.dtype, device=q.device)
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
    # The output geometry follows the value width (qk/value head dims may differ).
    expected_out = (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
    if out.shape != expected_out or out.dtype != q.dtype or dy.shape != expected_out or dy.dtype != q.dtype:
        raise ValueError("ISA attention saved output and gradient must match the value geometry")
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
    indexer_inputs = (
        ("idx_q", _IDX_Q),
        ("idx_k_c", _IDX_K),
        _IDX_MASK,
    )
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


# ---------------------------------------------------------------------------
# mHC: Sinkhorn doubly-stochastic projection of the branch mix matrix.  The
# reference expands exp/clamp plus 2*n_iters elementwise normalization rounds
# over the [Hc, Hc] mix matrix; the boundary declares that exact math (fp32
# internal, output rounding back to the input dtype) as one capture operator
# pair, so the backward phase sees one gradient boundary instead of the
# ~400-op expansion.  The gradient recomputes the forward trajectory and
# propagates the transposed normalization operators.
# ---------------------------------------------------------------------------
_MHC_MIX_AXES = ("branch_row", "branch_col")
_MHC_MIX = TensorMeaning("mix_matrix", _MHC_MIX_AXES, unknown_leading_axes=True)
_MHC_MIXED = TensorMeaning("normalized_mix", _MHC_MIX_AXES, unknown_leading_axes=True)


@torch.library.custom_op("system3_sim::mhc_sinkhorn", mutates_args=())
def mhc_sinkhorn(w: torch.Tensor, n_iters: int, clamp_exp: float, clamp_denom: float) -> torch.Tensor:
    raise RuntimeError("system3 virtual mhc_sinkhorn is available only for Meta simulation")


@mhc_sinkhorn.register_fake
def _mhc_sinkhorn_meta(w, n_iters, clamp_exp, clamp_denom):
    if w.ndim < 2 or w.shape[-1] != w.shape[-2] or min(*w.shape[-2:]) <= 0:
        raise ValueError("mhc_sinkhorn requires a positive square trailing mix matrix")
    if w.dtype not in _FLOAT_DTYPES:
        raise ValueError("mhc_sinkhorn requires FP16, BF16 or FP32 mix matrices")
    if type(n_iters) is not int or n_iters < 1:
        raise ValueError("mhc_sinkhorn requires a positive iteration count")
    if not (clamp_exp > 0) or not math.isfinite(clamp_exp):
        raise ValueError("mhc_sinkhorn requires a finite positive exp clamp")
    if not (clamp_denom > 0) or not math.isfinite(clamp_denom):
        raise ValueError("mhc_sinkhorn requires a finite positive denominator clamp")
    return torch.empty_like(w)


@torch.library.custom_op("system3_sim::mhc_sinkhorn_backward", mutates_args=())
def mhc_sinkhorn_backward(
    w: torch.Tensor,
    dy: torch.Tensor,
    n_iters: int,
    clamp_exp: float,
    clamp_denom: float,
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual mhc_sinkhorn backward is available only for Meta simulation")


@mhc_sinkhorn_backward.register_fake
def _mhc_sinkhorn_backward_meta(w, dy, n_iters, clamp_exp, clamp_denom):
    if w.shape != dy.shape or dy.dtype != w.dtype:
        raise ValueError("mhc_sinkhorn gradient must match the mix matrix's shape and precision")
    if not (clamp_exp > 0) or not math.isfinite(clamp_exp):
        raise ValueError("mhc_sinkhorn requires a finite positive exp clamp")
    if not (clamp_denom > 0) or not math.isfinite(clamp_denom):
        raise ValueError("mhc_sinkhorn requires a finite positive denominator clamp")
    return [torch.empty_like(w)]


def _mhc_sinkhorn_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])
    ctx.n_iters = inputs[1]
    ctx.clamp_exp = inputs[2]
    ctx.clamp_denom = inputs[3]


def _mhc_sinkhorn_backward(ctx, dy):
    (w,) = ctx.saved_tensors
    return (
        *mhc_sinkhorn_backward(w, dy, ctx.n_iters, ctx.clamp_exp, ctx.clamp_denom),
        None,
        None,
        None,
    )


mhc_sinkhorn.register_autograd(_mhc_sinkhorn_backward, setup_context=_mhc_sinkhorn_setup)


@torch.library.custom_op("system3_sim::engram_hash", mutates_args=())
def engram_hash(ids: torch.Tensor, multipliers: torch.Tensor, capacity: int, order: int) -> torch.Tensor:
    raise RuntimeError("system3 virtual engram_hash is available only for Meta simulation")


@engram_hash.register_fake
def _engram_hash_meta(ids, multipliers, capacity, order):
    if ids.ndim != 2 or ids.dtype != torch.int64 or min(ids.shape) <= 0:
        raise ValueError("engram_hash requires positive [batch, sequence] int64 token ids")
    if multipliers.ndim != 1 or multipliers.dtype != torch.int64 or multipliers.shape[0] <= 0:
        raise ValueError("engram_hash requires a positive int64 per-head multiplier vector")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("engram_hash requires a positive table capacity")
    if type(order) is not int or order < 1:
        raise ValueError("engram_hash requires a positive n-gram order")
    return torch.empty((*ids.shape, multipliers.shape[0]), dtype=torch.int64, device=ids.device)


_ENGRAM_IDS_AXES = ("batch", "sequence")
_ENGRAM_IDS = TensorMeaning("token_ids", _ENGRAM_IDS_AXES)
_ENGRAM_MULTIPLIERS = TensorMeaning("hash_multipliers", ("engram_head",))
_ENGRAM_KEYS = TensorMeaning("probe_keys", (*_ENGRAM_IDS_AXES, "engram_head"))


def _register_engram_hash_semantics() -> None:
    register_tensor_semantics(
        str(engram_hash._opoverload),
        OperatorTensorSemantics((("ids", _ENGRAM_IDS), ("multipliers", _ENGRAM_MULTIPLIERS)), (_ENGRAM_KEYS,)),
    )


@torch.library.custom_op("system3_sim::engram_gate", mutates_args=())
def engram_gate(
    x: torch.Tensor,
    table: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    out_weight: torch.Tensor,
    ids: torch.Tensor,
    epsilon: float,
    scale: float,
) -> torch.Tensor:
    raise RuntimeError("system3 virtual engram_gate is available only for Meta simulation")


@engram_gate.register_fake
def _engram_gate_meta(x, table, value, q_weight, k_weight, out_weight, ids, epsilon, scale):
    if x.shape != table.shape or x.shape != value.shape or min(x.shape) <= 0:
        raise ValueError("engram_gate requires matching positive projected query/key/value streams")
    for stream in (x, table, value):
        if stream.dtype not in _FLOAT_DTYPES:
            raise ValueError("engram_gate requires FP16, BF16 or FP32 projected streams")
    dim = x.shape[-1]
    for weight in (q_weight, k_weight, out_weight):
        if weight.shape != (dim,) or weight.dtype not in _FLOAT_DTYPES:
            raise ValueError("engram_gate norm weights must be one floating feature weight per stream feature")
    if ids.dtype != torch.int64 or ids.shape != x.shape[:-1]:
        raise ValueError("engram_gate requires int64 token ids matching the stream's token axes")
    if not (epsilon > 0) or not math.isfinite(epsilon):
        raise ValueError("engram_gate requires a finite positive norm epsilon")
    if not (scale > 0) or not math.isfinite(scale):
        raise ValueError("engram_gate requires a finite positive dot-product scale")
    # The reference gate math promotes the scaled dot and the gated product to
    # the widest stream precision (fp32 host table vs bf16 stream).
    promoted = torch.promote_types(x.dtype, torch.promote_types(table.dtype, value.dtype))
    return torch.empty(x.shape, dtype=promoted, device=x.device)


@torch.library.custom_op("system3_sim::engram_gate_backward", mutates_args=())
def engram_gate_backward(
    x: torch.Tensor,
    table: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    out_weight: torch.Tensor,
    ids: torch.Tensor,
    dy: torch.Tensor,
    epsilon: float,
    scale: float,
) -> list[torch.Tensor]:
    raise RuntimeError("system3 virtual engram_gate backward is available only for Meta simulation")


@engram_gate_backward.register_fake
def _engram_gate_backward_meta(x, table, value, q_weight, k_weight, out_weight, ids, dy, epsilon, scale):
    promoted = torch.promote_types(x.dtype, torch.promote_types(table.dtype, value.dtype))
    if dy.shape != x.shape or dy.dtype != promoted:
        raise ValueError("engram_gate gradient must match the gated output's shape and precision")
    return [
        torch.empty_like(x),
        torch.empty_like(table),
        torch.empty_like(value),
        torch.empty_like(q_weight),
        torch.empty_like(k_weight),
        torch.empty_like(out_weight),
    ]


def _engram_gate_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:7])
    ctx.epsilon = inputs[7]
    ctx.scale = inputs[8]


def _engram_gate_backward(ctx, dy):
    return (
        *engram_gate_backward(*ctx.saved_tensors, dy, ctx.epsilon, ctx.scale),
        None,
        None,
        None,
    )


engram_gate.register_autograd(_engram_gate_backward, setup_context=_engram_gate_setup)


_ENGRAM_STREAM_AXES = ("feature",)
_ENGRAM_STREAM = TensorMeaning("gate_stream", _ENGRAM_STREAM_AXES, unknown_leading_axes=True)
_ENGRAM_PROJECTED_KEY_QUERY = TensorMeaning("projected_key_query", _ENGRAM_STREAM_AXES, unknown_leading_axes=True)
_ENGRAM_PROJECTED_VALUE = TensorMeaning("projected_value", _ENGRAM_STREAM_AXES, unknown_leading_axes=True)
_ENGRAM_NORM_WEIGHT = TensorMeaning("norm_weight", _ENGRAM_STREAM_AXES)
_ENGRAM_GATED_OUTPUT = TensorMeaning("gated_output", _ENGRAM_STREAM_AXES, unknown_leading_axes=True)
_ENGRAM_STREAM_GRADIENT = TensorMeaning("gate_stream_gradient", _ENGRAM_STREAM_AXES, unknown_leading_axes=True)
_ENGRAM_KEY_QUERY_GRADIENT = TensorMeaning(
    "projected_key_query_gradient", _ENGRAM_STREAM_AXES, unknown_leading_axes=True
)
_ENGRAM_VALUE_GRADIENT = TensorMeaning("projected_value_gradient", _ENGRAM_STREAM_AXES, unknown_leading_axes=True)
_ENGRAM_NORM_WEIGHT_GRADIENT = TensorMeaning("norm_weight_gradient", _ENGRAM_STREAM_AXES)


def _register_engram_gate_semantics() -> None:
    register_tensor_semantics(
        str(engram_gate._opoverload),
        OperatorTensorSemantics(
            (
                ("x", _ENGRAM_STREAM),
                ("table", _ENGRAM_PROJECTED_KEY_QUERY),
                ("value", _ENGRAM_PROJECTED_VALUE),
                ("q_weight", _ENGRAM_NORM_WEIGHT),
                ("k_weight", _ENGRAM_NORM_WEIGHT),
                ("out_weight", _ENGRAM_NORM_WEIGHT),
                ("ids", _ENGRAM_IDS),
            ),
            (_ENGRAM_GATED_OUTPUT,),
        ),
    )
    register_tensor_semantics(
        str(engram_gate_backward._opoverload),
        OperatorTensorSemantics(
            (
                ("x", _ENGRAM_STREAM),
                ("table", _ENGRAM_PROJECTED_KEY_QUERY),
                ("value", _ENGRAM_PROJECTED_VALUE),
                ("q_weight", _ENGRAM_NORM_WEIGHT),
                ("k_weight", _ENGRAM_NORM_WEIGHT),
                ("out_weight", _ENGRAM_NORM_WEIGHT),
                ("ids", _ENGRAM_IDS),
                ("dy", _ENGRAM_GATED_OUTPUT),
            ),
            (
                _ENGRAM_STREAM_GRADIENT,
                _ENGRAM_KEY_QUERY_GRADIENT,
                _ENGRAM_VALUE_GRADIENT,
                _ENGRAM_NORM_WEIGHT_GRADIENT,
                _ENGRAM_NORM_WEIGHT_GRADIENT,
                _ENGRAM_NORM_WEIGHT_GRADIENT,
            ),
        ),
    )


_register_engram_hash_semantics()
_register_engram_gate_semantics()


def _register_mhc_semantics() -> None:
    register_tensor_semantics(
        str(mhc_sinkhorn._opoverload), OperatorTensorSemantics((("w", _MHC_MIX),), (_MHC_MIXED,))
    )
    output_gradient = TensorMeaning("output_gradient", _MHC_MIX.axes, _MHC_MIX.unknown_leading_axes)
    mix_gradient = TensorMeaning("mix_matrix_gradient", _MHC_MIX.axes, _MHC_MIX.unknown_leading_axes)
    register_tensor_semantics(
        str(mhc_sinkhorn_backward._opoverload),
        OperatorTensorSemantics((("w", _MHC_MIX), ("dy", output_gradient)), (mix_gradient,)),
    )


_register_mhc_semantics()


def install_system3_virtual_fusions(scope):
    """Replace the model-level fusion call sites with the virtual boundaries."""
    import torchtitan_npu.models.system3.model as system3_model
    from torchtitan_npu.models.system3.model import (
        EngramModule,
        GDN2Attention,
        IndexedSparseAttention,
        ISAIndexer,
        PolynomialRollingHash,
        SinkhornIteration,
    )

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

    def sinkhorn_projection(module, w):
        return mhc_sinkhorn(w, module.n_iters, 20.0, 1e-12)

    def ngram_hash(module, input_ids):
        # The reference Horner recurrence (h = (h*BASE + id*mult) % capacity,
        # per-head multipliers, order-window shifts) as one integer-mix
        # boundary matching the inference engram hash family.
        return engram_hash(input_ids, module.multipliers, module.capacity, module.order)

    def engram_gate_fusion(module, x_proj, table_proj, value_proj, input_ids):
        # The reference rmsnorm_ssr gate segment: per-stream float32 RMSNorm,
        # scaled query/key dot, signed-square-root sigmoid gate, static
        # id != 0 token mask, gated value stream. Matches the inference
        # projected_rms_ssr engram gate family (projections stay eager).
        return engram_gate(
            x_proj,
            table_proj,
            value_proj,
            module.q_norm.weight,
            module.k_norm.weight,
            module.output_norm.weight,
            input_ids,
            module.q_norm.eps,
            module.scale,
        )

    def rotary_fusion(q, cos, sin):
        # The eager half-split rotation (rotate_half: chunk into halves,
        # swap with sign) is npu_rotary_mul's native "half" mode: one
        # factor element per rotated pair. The reference cache concatenates
        # the pair factors twice (front == back), so slicing the front half
        # is numerically identical; reshape to the activation's rank for the
        # [B, S, H, D] broadcast. The reference casts the fp32 cache to the
        # stream dtype before the elementwise product; the bridge keeps it.
        half = q.shape[-1] // 2
        cos_b = cos[..., :half].to(q.dtype).reshape(1, cos.shape[0], 1, half)
        sin_b = sin[..., :half].to(q.dtype).reshape(1, sin.shape[0], 1, half)
        return torch_npu.npu_rotary_mul(q, cos_b, sin_b, rotary_mode="half")

    scope.replace(system3_model, "swiglu", activation)
    scope.replace(system3_model, "apply_rotary_to_q", rotary_fusion)
    scope.replace(GDN2Attention, "_delta_rule_scan", scan)
    scope.replace(ISAIndexer, "_select_topk", indexer_topk)
    scope.replace(IndexedSparseAttention, "_sparse_attention", selected_attention)
    scope.replace(SinkhornIteration, "forward", sinkhorn_projection)
    scope.replace(PolynomialRollingHash, "forward", ngram_hash)
    scope.replace(EngramModule, "_gate", engram_gate_fusion)
