# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shims for SMLA's SparseAttention/LiCompute/LiLoss (`torchtitan_npu.converters.
kernels.npu_smla`'s NpuSparseAttention/NpuLiCompute/NpuLiLoss in production). Records the real
production ACLNN op names (`aclnn.npu_sparse_attn_sharedkv` etc., verified against the real
ops/aclnn/*/binding.cpp C++ sources) into the active OpDispatchCapture, with analytically-derived
shapes -- never invoking the real JIT-compiled ACLNN extensions. See design doc for exact shape
formulas: docs/superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md."""

from __future__ import annotations

import torch

from torchtitan_npu.converters.kernels.npu_smla import _add_offset_to_valid_sparse_indices
from torchtitan_npu.models.deepseek_v4.model import LiCompute, LiLoss, SparseAttention
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


class _SimSparseAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, module_path):  # noqa: ANN001
        B, S, N, D = query.shape
        dtype = query.dtype

        metadata = torch.empty(1024, dtype=torch.int32, device=query.device)
        _record("aclnn.npu_sparse_attn_sharedkv_metadata", [query], [metadata], module_path)

        result = torch.empty((B, S, N, D), dtype=dtype, device=query.device)
        softmax_lse = torch.empty((B, S, N, 1), dtype=torch.float32, device=query.device)
        fwd_inputs = [query, ori_kv, sinks, metadata]
        if cmp_kv is not None:
            fwd_inputs.append(cmp_kv)
        if cmp_sparse_indices is not None:
            fwd_inputs.append(cmp_sparse_indices)
        _record("aclnn.npu_sparse_attn_sharedkv", fwd_inputs, [result, softmax_lse], module_path)

        ctx.save_for_backward(query, ori_kv, cmp_kv, result, softmax_lse, sinks)
        ctx.module_path = module_path
        return result

    @staticmethod
    def backward(ctx, grad_result):  # noqa: ANN001
        query, ori_kv, cmp_kv, result, softmax_lse, sinks = ctx.saved_tensors
        dquery = torch.empty_like(query)
        dori_kv = torch.empty_like(ori_kv)
        dsinks = torch.empty_like(sinks)
        dcmp_kv = torch.empty_like(cmp_kv) if cmp_kv is not None else None

        bwd_inputs = [query, ori_kv, result, softmax_lse, sinks, grad_result]
        if cmp_kv is not None:
            bwd_inputs.append(cmp_kv)
        bwd_outputs = [dquery, dori_kv, dsinks]
        if dcmp_kv is not None:
            bwd_outputs.append(dcmp_kv)
        _record("aclnn.npu_sparse_attn_sharedkv_grad", bwd_inputs, bwd_outputs, ctx.module_path)

        return dquery, dori_kv, dcmp_kv, None, dsinks, None


class SimNpuSparseAttention(SparseAttention):
    """Drop-in simulator replacement for `NpuSparseAttention`
    (`torchtitan_npu.converters.kernels.npu_smla`). Same forward() signature as base
    `SparseAttention`; never runs the real JIT-compiled ACLNN extension, only records the real
    op names + analytically-correct shapes.

    `__init__` mirrors `NpuSparseAttention.__init__(self, parent: SparseAttention)`'s exact
    `__dict__` shallow-copy pattern (`npu_smla.py:1330-1338`) -- required both for consistency
    and to satisfy torchtitan's `verify_module_protocol()` (subclassing `SparseAttention`
    transitively satisfies the `isinstance(mod, Module)` check with zero extra code)."""

    def __init__(self, parent: "SparseAttention") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, query_states, kv_states, attn_sink, kv_compress=None, compress_topk_idxs=None):
        if compress_topk_idxs is not None and compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)
        ori_kv = kv_states.unsqueeze(2).contiguous()
        cmp_kv = kv_compress.unsqueeze(2).contiguous() if kv_compress is not None else None
        cmp_sparse_indices = compress_topk_idxs.unsqueeze(2).contiguous() if self.compress_ratio == 4 else None
        module_path = self.__class__.__name__
        result = _SimSparseAttnFn.apply(
            query_states.contiguous(), ori_kv, cmp_kv, cmp_sparse_indices, attn_sink.float(), module_path
        )
        return result.contiguous()


class _SimLightningIndexerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, weights, index_topk, module_path):  # noqa: ANN001
        B, S, N_i, D_i = query.shape
        K = index_topk
        sparse_indices = torch.empty((B, S, 1, K), dtype=torch.int32, device=query.device)
        sparse_values = torch.empty((B, S, 1, K), dtype=query.dtype, device=query.device)
        _record("aclnn.npu_lightning_indexer", [query, key, weights], [sparse_indices, sparse_values], module_path)
        return sparse_indices.squeeze(2), sparse_values.squeeze(2)

    @staticmethod
    def backward(ctx, grad_topk_idxs, grad_index_score):  # noqa: ANN001
        # The real non-A5 npu_lightning_indexer call has no autograd kernel (see design doc
        # §4.2): compress_topk_idxs is non-differentiable (int32), and index_score's real
        # gradient flows through the separate LiLoss path via detached tensors, never back
        # through LiCompute. Mirrors the A5 LightningIndexer.backward's all-None return.
        return None, None, None, None, None


class SimNpuLiCompute(LiCompute):
    """Drop-in simulator replacement for `NpuLiCompute`
    (`torchtitan_npu.converters.kernels.npu_smla`). The real non-A5 implementation calls
    `_li_op.npu_lightning_indexer` directly with no `torch.autograd.Function` wrapper --
    `_SimLightningIndexerFn` adds one here purely for gradient-graph connectivity and
    consistent backward-phase tagging, matching the pattern established for `SimHcHead` in
    the MHC plan."""

    def __init__(self, parent: "LiCompute") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, q_indexer: torch.Tensor, k_indexer: torch.Tensor, weights: torch.Tensor, seqlen: int, offset: int):
        q_indexer = q_indexer.to(torch.bfloat16)
        k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
        weights = weights.to(torch.bfloat16)
        module_path = self.__class__.__name__
        compress_topk_idxs, index_score = _SimLightningIndexerFn.apply(q_indexer, k_indexer, weights, self.index_topk, module_path)
        compress_topk_idxs = _add_offset_to_valid_sparse_indices(compress_topk_idxs, offset)
        return compress_topk_idxs, index_score
