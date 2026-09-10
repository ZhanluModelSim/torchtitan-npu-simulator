# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only simulator shims for ar_llm model-specific fused ops.

All raw op names and input signatures are REUSED from the DeepSeek-V4 SMLA/MHC
shim set (``smla_shim.py`` / ``mhc_shim.py``) so downstream op mapping and
shape parsing stay unified:

- Sparse attention (both CSA and HCA cores): ``aclnn.npu_sparse_attn_sharedkv``
  with the DSV4 input-count convention -- the parser distinguishes variants by
  tensor count:
      [query, ori_kv, sinks, metadata]                      (4, no compression)
      [query, ori_kv, sinks, metadata, cmp_kv]              (5, CSA: strided KV)
      [..., cmp_kv, cmp_sparse_indices]                     (6, HCA: indexer topk)
  Each forward also records ``aclnn.npu_sparse_attn_sharedkv_metadata`` and
  outputs ``[result, softmax_lse]`` like DSV4; backward records
  ``aclnn.npu_sparse_attn_sharedkv_grad`` with the same packing.
- Lightning indexer: forward ``aclnn.npu_lightning_indexer``
  [query, key, weights] -> [sparse_indices, sparse_values] with NO backward op
  (DSV4 non-A5 behavior); the indexer gradient is represented by
  ``aclnn.npu_sparse_lightning_indexer_grad_kl_loss`` in backward.
- mHC: ``triton._triton_hc_sinkhorn_comb_fwd/bwd_kernel`` for the Sinkhorn
  comb and ``triton.hc_pre_bmm_forward/backward`` for the channel mixing bmm
  (rms_norm/matmul steps run as real meta ops via the Sim RMSNorm converter
  and aten matmul, matching DSV4's decomposition).

Shims are bound after model construction and parallelization (same seam as
``apply_kimi_k3_shims``) so FQN, DTensor placements, and hooks stay intact.
Deviations from DSV4 signatures are shape-level only and documented inline:
ar_llm packs K/V into ``ori_kv``/``cmp_kv``'s dim-2 and has no indexer
``weights`` projection (recorded as ones).
"""

from __future__ import annotations

from types import MethodType

import torch

from torchtitan_npu.models.ar_llm.attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    KimiDeltaAttentionCore,
)
from torchtitan_npu.models.ar_llm.model import HyperConnectionBlock
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
from torchtitan_npu.simulator.hardware_shims.kda_shim import (
    _current_module_path,
    _record,
    _uncaptured_empty_like,
)

_KDA_SHIM_MARKER = "_simulator_ar_llm_kda_shim_installed"
_CSA_SHIM_MARKER = "_simulator_ar_llm_csa_shim_installed"
_HCA_SHIM_MARKER = "_simulator_ar_llm_hca_shim_installed"
_MHC_SHIM_MARKER = "_simulator_ar_llm_mhc_shim_installed"


def _uncaptured_empty(shape, dtype, device):  # noqa: ANN001
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=dtype, device=device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=dtype, device=device)


def _pack_kv(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pack K/V into DSV4's ``ori_kv``/``cmp_kv`` layout ``[B, T, 1, 2, N, D]``."""
    b, t, nh, hd = k.shape
    return _uncaptured_empty((b, t, 1, 2, nh, hd), k.dtype, k.device)


def _sinks_float(sink_bias: torch.Tensor | None, num_heads: int, device, dtype) -> torch.Tensor:
    if sink_bias is not None:
        return sink_bias.float()
    return torch.zeros(num_heads, dtype=dtype, device=device)


class _SimChunkKDAFn(torch.autograd.Function):
    """Shape-only bridge for the KDA chunk kernel on meta tensors (same raw
    op name as kimi_k3's chunk_kda; ar_llm's double-gate variant passes
    alpha/beta_erase/beta_write instead of g/A_log/dt_bias)."""

    @staticmethod
    def forward(ctx, q, k, v, alpha, beta_erase, beta_write, module_path):  # noqa: ANN001
        output = _uncaptured_empty_like(v)
        _record(
            "triton_ascend_kernels.chunk_kda",
            [q, k, v, alpha, beta_erase, beta_write],
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, alpha, beta_erase, beta_write)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, alpha, beta_erase, beta_write = ctx.saved_tensors
        grads = [_uncaptured_empty_like(t) for t in (q, k, v, alpha, beta_erase, beta_write)]
        _record(
            "triton_ascend_kernels.chunk_kda_grad",
            [q, k, v, alpha, beta_erase, beta_write, grad_output],
            grads,
            ctx.module_path,
        )
        return (*grads, None)


def sim_ar_llm_chunk_kda(module, q, k, v, alpha, beta_erase, beta_write):  # noqa: ANN001
    return _SimChunkKDAFn.apply(
        q, k, v, alpha, beta_erase, beta_write, _current_module_path()
    )


class _SimCSAFn(torch.autograd.Function):
    """CSA core: window + strided-global attention = DSV4's compressed-KV
    variant (5-input sharedkv, no sparse indices)."""

    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, sinks, module_path):  # noqa: ANN001
        b, s, nh, hd = query.shape
        metadata = _uncaptured_empty((1024,), torch.int32, query.device)
        _record("aclnn.npu_sparse_attn_sharedkv_metadata", [query], [metadata], module_path)
        result = _uncaptured_empty_like(query)
        softmax_lse = _uncaptured_empty((b, s, nh, 1), torch.float32, query.device)
        _record(
            "aclnn.npu_sparse_attn_sharedkv",
            [query, ori_kv, sinks, metadata, cmp_kv],
            [result, softmax_lse],
            module_path,
        )
        ctx.save_for_backward(query, ori_kv, cmp_kv, result, softmax_lse, sinks)
        ctx.module_path = module_path
        return result

    @staticmethod
    def backward(ctx, grad_result):  # noqa: ANN001
        query, ori_kv, cmp_kv, result, softmax_lse, sinks = ctx.saved_tensors
        d_query = _uncaptured_empty_like(query)
        d_ori_kv = _uncaptured_empty_like(ori_kv)
        d_cmp_kv = _uncaptured_empty_like(cmp_kv)
        d_sinks = _uncaptured_empty_like(sinks)
        _record(
            "aclnn.npu_sparse_attn_sharedkv_grad",
            [query, ori_kv, result, softmax_lse, sinks, grad_result, cmp_kv],
            [d_query, d_ori_kv, d_sinks, d_cmp_kv],
            ctx.module_path,
        )
        return d_query, d_ori_kv, d_cmp_kv, d_sinks, None


def _sim_csa_forward(module, q, k, v, sink_bias):  # noqa: ANN001
    b, s, nh, hd = q.shape
    query = q.contiguous()
    ori_kv = _pack_kv(k, v)
    cl = k.shape[1] // module.compress_ratio
    cmp_kv = _uncaptured_empty(
        (b, cl, 1, 2, nh, hd), q.dtype, q.device
    )
    sinks = _sinks_float(sink_bias, nh, q.device, torch.float32)
    return _SimCSAFn.apply(query, ori_kv, cmp_kv, sinks, _current_module_path())


class _SimLightningIndexerFn(torch.autograd.Function):
    """HCA indexer: DSV4's lightning indexer, forward record only (no
    backward op exists for npu_lightning_indexer; the gradient op is the
    sparse_lightning_indexer_grad_kl_loss recorded by the core backward)."""

    @staticmethod
    def forward(ctx, idx_q, idx_k_c, weights, topk, module_path):  # noqa: ANN001
        b, s = idx_q.shape[0], idx_q.shape[1]
        sparse_indices = _uncaptured_empty((b, s, 1, topk), torch.int32, idx_q.device)
        sparse_values = _uncaptured_empty((b, s, 1, topk), idx_q.dtype, idx_q.device)
        _record(
            "aclnn.npu_lightning_indexer",
            [idx_q, idx_k_c, weights],
            [sparse_indices, sparse_values],
            module_path,
        )
        return sparse_indices.squeeze(2), sparse_values.squeeze(2)

    @staticmethod
    def backward(ctx, grad_indices, grad_values):  # noqa: ANN001
        return None, None, None, None, None


class _SimHCACoreFn(torch.autograd.Function):
    """HCA core: indexer top-k + attention over selected KV = DSV4's
    topk-indexed variant (6-input sharedkv). Backward records the sharedkv
    grad plus DSV4's sparse_lightning_indexer_grad_kl_loss for the indexer
    gradients."""

    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, idx_q, idx_k_c, weights, sinks, topk, module_path):  # noqa: ANN001
        b, s, nh, hd = query.shape
        metadata = _uncaptured_empty((1024,), torch.int32, query.device)
        _record("aclnn.npu_sparse_attn_sharedkv_metadata", [query], [metadata], module_path)

        sparse_indices, _ = _SimLightningIndexerFn.apply(
            idx_q.to(torch.bfloat16), idx_k_c.to(torch.bfloat16), weights.to(torch.bfloat16), topk, module_path
        )
        sparse_indices = sparse_indices.unsqueeze(2).contiguous()

        result = _uncaptured_empty_like(query)
        softmax_lse = _uncaptured_empty((b, s, nh, 1), torch.float32, query.device)
        _record(
            "aclnn.npu_sparse_attn_sharedkv",
            [query, ori_kv, sinks, metadata, cmp_kv, sparse_indices],
            [result, softmax_lse],
            module_path,
        )
        ctx.save_for_backward(query, ori_kv, cmp_kv, idx_q, idx_k_c, weights, sparse_indices, result, softmax_lse, sinks)
        ctx.module_path = module_path
        return result

    @staticmethod
    def backward(ctx, grad_result):  # noqa: ANN001
        query, ori_kv, cmp_kv, idx_q, idx_k_c, weights, sparse_indices, result, softmax_lse, sinks = ctx.saved_tensors
        d_query = _uncaptured_empty_like(query)
        d_ori_kv = _uncaptured_empty_like(ori_kv)
        d_cmp_kv = _uncaptured_empty_like(cmp_kv)
        d_sinks = _uncaptured_empty_like(sinks)
        _record(
            "aclnn.npu_sparse_attn_sharedkv_grad",
            [query, ori_kv, result, softmax_lse, sinks, grad_result, cmp_kv],
            [d_query, d_ori_kv, d_sinks, d_cmp_kv],
            ctx.module_path,
        )
        d_idx_q = _uncaptured_empty_like(idx_q)
        d_idx_k_c = _uncaptured_empty_like(idx_k_c)
        d_weights = _uncaptured_empty_like(weights)
        loss = _uncaptured_empty((1,), torch.float32, query.device)
        _record(
            "aclnn.npu_sparse_lightning_indexer_grad_kl_loss",
            [query, cmp_kv, idx_q, idx_k_c, weights, sparse_indices],
            [d_idx_q, d_idx_k_c, d_weights, loss],
            ctx.module_path,
        )
        return d_query, d_ori_kv, d_cmp_kv, d_idx_q, d_idx_k_c, d_weights, d_sinks, None, None


def _sim_hca_forward(module, q, k, v, hidden_states, sink_bias):  # noqa: ANN001
    b, s, nh, hd = q.shape
    query = q.contiguous()
    ori_kv = _pack_kv(k, v)
    cl = k.shape[1] // module.compress_ratio
    cmp_kv = _uncaptured_empty((b, cl, 1, 2, nh, hd), q.dtype, q.device)
    idx_q = module.indexer_q_norm(module.indexer_q(hidden_states))
    idx_k_c = module.indexer_k_norm(module.indexer_k(hidden_states))[:, :: module.compress_ratio]
    weights = _uncaptured_empty((b, s, 1, 1), torch.float32, q.device)
    sinks = _sinks_float(sink_bias, nh, q.device, torch.float32)
    return _SimHCACoreFn.apply(
        query, ori_kv, cmp_kv, idx_q, idx_k_c, weights, sinks, module.topk, _current_module_path()
    )


class _SimSinkhornCombFn(torch.autograd.Function):
    """mHC Sinkhorn comb (ar_llm's SinkhornIteration over the [hc, hc] mix
    weights) recorded with DSV4's sinkhorn_comb kernel name."""

    @staticmethod
    def forward(ctx, mix_weights, module_path):  # noqa: ANN001
        mixed = _uncaptured_empty_like(mix_weights)
        _record(
            "triton._triton_hc_sinkhorn_comb_fwd_kernel",
            [mix_weights],
            [mixed],
            module_path,
        )
        ctx.save_for_backward(mix_weights)
        ctx.module_path = module_path
        return mixed

    @staticmethod
    def backward(ctx, grad_mixed):  # noqa: ANN001
        (mix_weights,) = ctx.saved_tensors
        grad_weights = _uncaptured_empty_like(mix_weights)
        _record(
            "triton._triton_hc_sinkhorn_comb_bwd_kernel",
            [grad_mixed, mix_weights],
            [grad_weights],
            ctx.module_path,
        )
        return grad_weights, None


class _SimHcBmmFn(torch.autograd.Function):
    """mHC channel mixing (ar_llm's expand-channel einsum with the comb
    weights) recorded with DSV4's hc_pre_bmm kernel name."""

    @staticmethod
    def forward(ctx, mix, expanded, module_path):  # noqa: ANN001
        mixed = _uncaptured_empty_like(expanded)
        _record(
            "triton.hc_pre_bmm_forward",
            [mix, expanded],
            [mixed],
            module_path,
        )
        ctx.save_for_backward(mix, expanded)
        ctx.module_path = module_path
        return mixed

    @staticmethod
    def backward(ctx, grad_mixed):  # noqa: ANN001
        mix, expanded = ctx.saved_tensors
        grad_mix = _uncaptured_empty_like(mix)
        grad_expanded = _uncaptured_empty_like(expanded)
        _record(
            "triton.hc_pre_bmm_backward",
            [mix, expanded, grad_mixed],
            [grad_mix, grad_expanded],
            ctx.module_path,
        )
        return grad_mix, grad_expanded, None


def _sim_hc_block_forward(module, x):  # noqa: ANN001
    b, s, d = x.shape
    residual = x
    xn = module.pre_norm(x)
    expanded = module.expand(xn).view(b, s, module.hc_mult, d)
    mix = _SimSinkhornCombFn.apply(module.mix_weights, _current_module_path())
    mixed = _SimHcBmmFn.apply(mix, expanded, _current_module_path())
    merged = mixed.reshape(b, s, module.hc_mult * d)
    return residual + module.post_norm(module.contract(merged))


def apply_ar_llm_shims(model) -> None:
    """Bind ar_llm shape-only shims while preserving module hooks."""
    for module in model.modules():
        if isinstance(module, KimiDeltaAttentionCore):
            if not getattr(module, _KDA_SHIM_MARKER, False):
                module._chunk_kda = MethodType(sim_ar_llm_chunk_kda, module)
                setattr(module, _KDA_SHIM_MARKER, True)
        elif isinstance(module, CompressedSparseAttention):
            if not getattr(module, _CSA_SHIM_MARKER, False):
                module.forward = MethodType(_sim_csa_forward, module)
                setattr(module, _CSA_SHIM_MARKER, True)
        elif isinstance(module, HeavilyCompressedAttention):
            if not getattr(module, _HCA_SHIM_MARKER, False):
                module.forward = MethodType(_sim_hca_forward, module)
                setattr(module, _HCA_SHIM_MARKER, True)
        elif isinstance(module, HyperConnectionBlock):
            if not getattr(module, _MHC_SHIM_MARKER, False):
                module.forward = MethodType(_sim_hc_block_forward, module)
                setattr(module, _MHC_SHIM_MARKER, True)
