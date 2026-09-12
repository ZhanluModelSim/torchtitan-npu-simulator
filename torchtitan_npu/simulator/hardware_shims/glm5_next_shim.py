# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only simulator shims for glm5_next model-specific fused ops.

Records the real production op names into the active OpDispatchCapture with
analytically-correct shapes. The op interfaces follow
``hardware_shims/OP_INTERFACE_REFERENCE.md`` (the contract the downstream
cost model parses):

- KDA core:              ``triton_ascend_kernels.chunk_kda`` (fwd 5 inputs)
                         ``triton_ascend_kernels.chunk_kda_grad`` (bwd inputs
                         ``[q, k, v, g, beta, grad_output]``, 5 same-shape grads)
                         (the fused qkv short conv stays a real aten conv1d,
                         same as kimi_k3 -- no causal_conv1d op is modeled)
- DSA indexer selection: ``aclnn.npu_lightning_indexer``
                         inputs ``[query_idx [B,S,N_idx,D_idx],
                         key_idx [B,cl,1,D_idx], weights [B,S,N_idx]]`` ->
                         ``[sparse_indices [B,S,1,K] int32, sparse_values]``;
                         no backward op (frozen indexer, no autograd node)
- DSA sparse attention:  ``aclnn.npu_sparse_attn_sharedkv_metadata`` +
                         6-input ``aclnn.npu_sparse_attn_sharedkv``
                         (``[query, ori_kv, sinks, metadata, cmp_kv,
                         cmp_sparse_indices]`` -> ``[result, softmax_lse]``);
                         backward 7 inputs -> ``[d_query, d_ori_kv, d_sinks,
                         d_cmp_kv]``

Placeholders follow reference rule §0.5: GLM has no attention sink
(``zeros[N]`` f32), no KV compression (ratio=1: ``cmp_kv`` mirrors
``ori_kv`` so the 6-input top-k variant applies), and head-collapsed
``ori_kv`` uses the per-head-KV convention ``nh*(k_dim+v_dim)``.

mHC pre/post are covered by the shared DSv4 ``npu_mhc_pre/post`` converters
plus ``apply_mhc_shims`` and need no extra binding here.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.glm5_next.attention import GlmDeltaAttention, GlmDsaAttention
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
from torchtitan_npu.simulator.hardware_shims.kda_shim import (
    _current_module_path,
    _record,
    _uncaptured_empty_like,
)

_KDA_SHIM_MARKER = "_simulator_glm5_next_kda_shim_installed"
_DSA_SHIM_MARKER = "_simulator_glm5_next_dsa_shim_installed"


def _uncaptured_empty(shape, dtype, device):  # noqa: ANN001
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=dtype, device=device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=dtype, device=device)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class _SimChunkKDAFn(torch.autograd.Function):
    """Shape-only bridge for the KDA chunk kernel on meta tensors.

    glm5_next pre-computes the forget gate ``g`` (decay/A_log/lower-bound live
    in the gate module), so the kernel receives g directly. Backward records
    the cost-model interface ``[q, k, v, g, beta, grad_output]``.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, module_path):  # noqa: ANN001
        output = _uncaptured_empty_like(v)
        _record(
            "triton_ascend_kernels.chunk_kda",
            [q, k, v, g, beta],
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, g, beta = ctx.saved_tensors
        grads = [_uncaptured_empty_like(t) for t in (q, k, v, g, beta)]
        _record(
            "triton_ascend_kernels.chunk_kda_grad",
            [q, k, v, g, beta, grad_output],
            grads,
            ctx.module_path,
        )
        return (*grads, None)


def _sim_chunk_kda(module, q, k, v, g, beta):  # noqa: ANN001
    return _SimChunkKDAFn.apply(q, k, v, g, beta, _current_module_path())


class _SimDsaIndexerFn(torch.autograd.Function):
    """k-pool compressed indexer scoring + top-k selection, fused op record.

    Interface (OP_INTERFACE_REFERENCE.md §3): inputs
    ``[query_idx [B,S,N_idx,D_idx], key_idx [B,cl,1,D_idx], weights [B,S,N_idx]]``
    -> outputs ``[sparse_indices [B,S,1,K] int32, sparse_values [B,S,1,K]]``.
    ``K`` is the *effective* top-k at pool level:
    ``min(topk // kpool, cl)``. No backward op: the glm5_next indexer is
    frozen and invoked under ``torch.no_grad`` (no autograd node exists).
    """

    @staticmethod
    def forward(ctx, query_idx, key_idx, weights, select_pools, module_path):  # noqa: ANN001
        b, s = query_idx.shape[0], query_idx.shape[1]
        sparse_indices = _uncaptured_empty((b, s, 1, select_pools), torch.int32, query_idx.device)
        sparse_values = _uncaptured_empty((b, s, 1, select_pools), query_idx.dtype, query_idx.device)
        _record(
            "aclnn.npu_lightning_indexer",
            [query_idx, key_idx, weights],
            [sparse_indices, sparse_values],
            module_path,
        )
        return sparse_indices.squeeze(2), sparse_values.squeeze(2)

    @staticmethod
    def backward(ctx, grad_indices, grad_values):  # noqa: ANN001
        # The real non-A5 lightning indexer has no autograd kernel; the
        # glm5_next indexer is additionally frozen (MODEL_CONTRACT.md §4.2).
        return None, None, None, None, None


class _SimDsaCoreFn(torch.autograd.Function):
    """Sparse attention over the top-k union, fused op record.

    Interface (OP_INTERFACE_REFERENCE.md §2, 6-input top-k variant): metadata
    op first, then main op ``[query, ori_kv, sinks, metadata, cmp_kv,
    cmp_sparse_indices]`` -> ``[result, softmax_lse]``; backward
    ``[query, ori_kv, result, softmax_lse, sinks, grad_result, cmp_kv]`` ->
    ``[d_query, d_ori_kv, d_sinks, d_cmp_kv]``.
    """

    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, module_path):  # noqa: ANN001
        b, s, n, d = query.shape

        metadata = _uncaptured_empty((1024,), torch.int32, query.device)
        _record("aclnn.npu_sparse_attn_sharedkv_metadata", [query], [metadata], module_path)

        result = _uncaptured_empty((b, s, n, d), query.dtype, query.device)
        softmax_lse = _uncaptured_empty((b, s, n, 1), torch.float32, query.device)
        _record(
            "aclnn.npu_sparse_attn_sharedkv",
            [query, ori_kv, sinks, metadata, cmp_kv, cmp_sparse_indices],
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
        d_sinks = _uncaptured_empty_like(sinks)
        d_cmp_kv = _uncaptured_empty_like(cmp_kv)
        _record(
            "aclnn.npu_sparse_attn_sharedkv_grad",
            [query, ori_kv, result, softmax_lse, sinks, grad_result, cmp_kv],
            [d_query, d_ori_kv, d_sinks, d_cmp_kv],
            ctx.module_path,
        )
        return d_query, d_ori_kv, d_cmp_kv, None, d_sinks, None


def _sim_dsa_forward(self, hidden_states, attention_masks=None, positions=None):  # noqa: ANN001
    del attention_masks, positions
    b, s, _ = hidden_states.shape
    module_path = _current_module_path()
    indexer = self.indexer

    # Real indexer projections (captured as aten ops, same as production).
    q_resid = self.q_a_norm(self.q_a_proj(hidden_states))
    query = self.q_b_proj(q_resid).view(b, s, self.num_heads, self.qk_nope_head_dim)
    kv_pass = self.kv_a_norm(self.kv_a_proj_with_mqa(hidden_states))
    kv = self.kv_b_proj(kv_pass).view(b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    key, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    # ---- lightning indexer (pool scoring + top-k), effective top-k only ----
    # Frozen indexer: projections and selection run under no_grad (no autograd
    # node, hence no invented backward op).
    with torch.no_grad():
        query_idx = self.indexer.wq_b(q_resid).view(b, s, indexer.n_heads, indexer.head_dim)
        weights = self.indexer.weights_proj(hidden_states)
        pools = s // indexer.kpool
        select_pools = min(indexer.topk // indexer.kpool, pools)
        # GLM scores compressed k-pool candidates: the head-folded indexer key
        # is the pooled key [B, pools, 1, D_idx] (pooling fused into the kernel).
        key_idx = _uncaptured_empty((b, pools, 1, indexer.head_dim), query_idx.dtype, query_idx.device)
        pool_indices, _ = _SimDsaIndexerFn.apply(query_idx, key_idx, weights, select_pools, module_path)

    # ---- sparse attention main op (6-input top-k variant, differentiable) ----
    # ori_kv: head-collapsed per-head KV (nh*(k_dim+v_dim)), reference §2.
    kv_dim = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
    ori_kv = _uncaptured_empty((b, s, 1, kv_dim), query.dtype, query.device)
    # GLM has no attention sink: zeros[N] f32 placeholder (rule §0.5).
    sinks = _uncaptured_empty((self.num_heads,), torch.float32, query.device)
    # ratio=1 (no KV compression): cmp_kv mirrors ori_kv so the 6-input
    # variant carries the pool top-k indices.
    cmp_kv = _uncaptured_empty_like(ori_kv)
    cmp_sparse_indices = pool_indices.unsqueeze(2)
    result = _SimDsaCoreFn.apply(query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, module_path)

    return self.o_proj(result.reshape(b, s, self.num_heads * self.v_head_dim))


def apply_glm5_next_shims(model) -> None:
    """Bind glm5_next shape-only shims while preserving module hooks.

    The fused qkv short conv is intentionally NOT shimmed: like kimi_k3's
    ``ShortConvolution`` it stays a real depthwise ``F.conv1d`` and is
    captured as ``aten.convolution.default`` (no ``causal_conv1d`` fused op
    exists in the modeled op set).
    """
    for module in model.modules():
        if isinstance(module, GlmDeltaAttention):
            if not getattr(module, _KDA_SHIM_MARKER, False):
                module._chunk_kda = MethodType(_sim_chunk_kda, module)
                setattr(module, _KDA_SHIM_MARKER, True)
        elif isinstance(module, GlmDsaAttention):
            if not getattr(module, _DSA_SHIM_MARKER, False):
                module.forward = MethodType(_sim_dsa_forward, module)
                setattr(module, _DSA_SHIM_MARKER, True)
