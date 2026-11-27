# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-only selected-key boundaries for K3.1, without changing device execution.

Q/K/V remain expanded per head, unlike absorbed inference DSA. A custom contract
keeps value width separate from score width and saves output/LSE for backward;
the legacy dense attention boundary has neither of these sparse contracts.
"""

import math

import torch

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)


@torch.library.custom_op("fused::dsa_indexer", mutates_args=())
def dsa_indexer(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, topk: int) -> torch.Tensor:
    raise RuntimeError("Kimi virtual DSA indexer is available only for Meta simulation")


@dsa_indexer.register_fake
def _indexer_meta(q, k, weights, topk):
    if q.ndim != 4 or k.shape != (q.shape[0], q.shape[1], q.shape[3]):
        raise ValueError("DSA indexer requires B/S/H/D query and B/S/D key")
    if weights.shape != q.shape[:-1] or weights.dtype != torch.float32 or topk < 1:
        raise ValueError("DSA indexer requires scaled FP32 weights and positive top-k")
    if min(q.shape) <= 0 or q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or k.dtype != q.dtype:
        raise ValueError("DSA indexer requires positive floating-point query/key geometry")
    if len({t.device for t in (q, k, weights)}) != 1:
        raise ValueError("DSA indexer operands must share placement")
    return torch.empty((*q.shape[:2], min(q.shape[1], topk)), dtype=torch.int64, device=q.device)


def _validate_attention(q, k, v, indices):
    if q.ndim != 4 or k.shape != q.shape or v.shape[:-1] != q.shape[:-1]:
        raise ValueError("DSA attention requires equal B/H/S dimensions with distinct QK/V widths")
    if min((*q.shape, v.shape[-1])) <= 0 or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("DSA attention requires positive floating-point geometry")
    if indices.ndim != 3 or indices.shape[:2] != (q.shape[0], q.shape[2]):
        raise ValueError("DSA indices must be B/S/K")
    if not 0 < indices.shape[2] <= q.shape[2] or indices.dtype != torch.int64:
        raise ValueError("DSA indices require positive bounded width and int64 storage")
    if len({t.dtype for t in (q, k, v)}) != 1 or len({t.device for t in (q, k, v, indices)}) != 1:
        raise ValueError("DSA attention tensors must share dtype and placement")


@torch.library.custom_op("fused::dsa_attention", mutates_args=())
def dsa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    raise RuntimeError("Kimi virtual selected attention is available only for Meta simulation")


@dsa_attention.register_fake
def _attention_meta(q, k, v, indices, scale):
    _validate_attention(q, k, v, indices)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("DSA attention requires finite positive scale")
    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=v.dtype, device=v.device)
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    return out, lse


@torch.library.custom_op("fused::dsa_attention_grad", mutates_args=())
def dsa_attention_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dy: torch.Tensor,
    scale: float,
) -> list[torch.Tensor]:
    raise RuntimeError("Kimi virtual selected attention gradient is available only for Meta simulation")


@dsa_attention_grad.register_fake
def _attention_grad_meta(q, k, v, indices, out, lse, dy, scale):
    _validate_attention(q, k, v, indices)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("DSA attention requires finite positive scale")
    if out.shape != (*q.shape[:-1], v.shape[-1]) or dy.shape != out.shape or lse.shape != q.shape[:-1]:
        raise ValueError("DSA saved output, LSE and output gradient disagree with the forward contract")
    if out.dtype != v.dtype or dy.dtype != out.dtype or lse.dtype != torch.float32:
        raise ValueError("DSA saved output and LSE dtypes disagree with forward")
    if any(t.device != q.device for t in (out, lse, dy)):
        raise ValueError("DSA saved tensors and output gradient must share placement")
    return [torch.empty_like(t) for t in (q, k, v)]


def _attention_setup(ctx, inputs, output):
    q, k, v, indices, ctx.scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, indices, out, lse)
    ctx.mark_non_differentiable(lse)


def _attention_backward(ctx, dy, unused_lse_gradient):
    return (*dsa_attention_grad(*ctx.saved_tensors, dy, ctx.scale), None, None)


dsa_attention.register_autograd(_attention_backward, setup_context=_attention_setup)

_Q = TensorMeaning("query", ("batch", "head", "q_len", "qk_head_dim"))
_K = TensorMeaning("key", ("batch", "head", "kv_len", "qk_head_dim"))
_V = TensorMeaning("value", ("batch", "head", "kv_len", "value_head_dim"))
_INDICES = TensorMeaning("selected_indices", ("batch", "q_len", "selected_key"))
_OUT = TensorMeaning("attention_output", ("batch", "head", "q_len", "value_head_dim"))
_LSE = TensorMeaning("logsumexp", ("batch", "head", "q_len"))
_INPUTS = (("q", _Q), ("k", _K), ("v", _V), ("indices", _INDICES))
register_tensor_semantics(
    str(dsa_indexer._opoverload),
    OperatorTensorSemantics(
        (
            ("q", TensorMeaning("query", ("batch", "sequence", "index_head", "index_head_dim"))),
            ("k", TensorMeaning("key", ("batch", "sequence", "index_head_dim"))),
            ("weights", TensorMeaning("index_weights", ("batch", "sequence", "index_head"))),
        ),
        (_INDICES,),
    ),
)
register_tensor_semantics(str(dsa_attention._opoverload), OperatorTensorSemantics(_INPUTS, (_OUT, _LSE)))
register_tensor_semantics(
    str(dsa_attention_grad._opoverload),
    OperatorTensorSemantics(
        (*_INPUTS, ("out", _OUT), ("lse", _LSE), ("dy", TensorMeaning("output_gradient", _OUT.axes))),
        tuple(
            TensorMeaning(role + "_gradient", meaning.axes)
            for role, meaning in (("query", _Q), ("key", _K), ("value", _V))
        ),
    ),
)


def install_kimi_dsa_virtual_fusions(scope):
    """Scope only selection/core, leaving projections and normalizations visible."""
    from torchtitan_npu.models.kimi_k3_1 import attention as owner
    from torchtitan_npu.models.kimi_k3_1.dsa import KimiDSAIndexer

    def select(module, queries, keys, weights):
        return dsa_indexer(queries, keys, weights, module.topk)

    def selected_attention(query, key, value, indices, scale):
        out, _ = dsa_attention(query, key, value, indices, scale)
        return out

    scope.replace(KimiDSAIndexer, "_select", select)
    scope.replace(owner, "selected_key_attention", selected_attention)
