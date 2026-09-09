# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only simulator shims for ar_llm model-specific fused ops.

Records the real production op names (see MODEL_CONTRACT.md section 9) into
the active OpDispatchCapture with analytically-correct shapes:

- KDA core:              ``triton_ascend_kernels.chunk_kda[_grad]``
- CSA core:              ``aclnn.npu_sparse_attn_sharedkv[_grad]``
- HCA indexer + core:    ``aclnn.npu_lightning_indexer[_grad]`` +
                         ``aclnn.npu_sparse_attn_sharedkv[_grad]``
- mHC Sinkhorn mix:      ``triton._triton_hc_sinkhorn_[fwd|bwd]_kernel``

Shims are bound after model construction and parallelization (same seam as
``apply_kimi_k3_shims``) so FQN, DTensor placements, and hooks stay intact.
The per-head ``attn_sink`` bias participates in the forward record but is
omitted from the backward record (zero-gradient, negligible byte volume).
The LatentMoE GMM path needs no shim: its real forward is ``aten._grouped_mm``
calls, which are captured directly.
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


def _current_module_path() -> str:
    capture = get_active_capture()
    if capture is not None and capture.module_path_tracker is not None:
        return capture.module_path_tracker.current_path()
    return ""


class _SimChunkKDAFn(torch.autograd.Function):
    """Shape-only bridge for the KDA chunk kernel on meta tensors."""

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
    """Window + strided-global attention as one fused op."""

    @staticmethod
    def forward(ctx, q, k, v, sink_bias, module_path):  # noqa: ANN001
        output = _uncaptured_empty_like(v)
        inputs = [q, k, v] + ([sink_bias] if sink_bias is not None else [])
        _record("aclnn.npu_sparse_attn_sharedkv", inputs, [output], module_path)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        empty = _uncaptured_empty_like(grad_output)
        _record(
            "aclnn.npu_sparse_attn_sharedkv_grad",
            [grad_output],
            [empty, empty, empty],
            ctx.module_path,
        )
        return empty, empty, empty, None, None


def _sim_csa_forward(module, q, k, v, sink_bias):  # noqa: ANN001
    return _SimCSAFn.apply(q, k, v, sink_bias, _current_module_path())


class _SimHCACoreFn(torch.autograd.Function):
    """Indexer top-k scoring + attention over selected KV as two fused ops."""

    @staticmethod
    def forward(ctx, q, k_c, v_c, idx_q, idx_k_c, sink_bias, module_path):  # noqa: ANN001
        scores = _uncaptured_empty(
            (q.shape[0], q.shape[1], k_c.shape[1]), q.dtype, q.device
        )
        _record("aclnn.npu_lightning_indexer", [idx_q, idx_k_c], [scores], module_path)
        output = _uncaptured_empty_like(q)
        inputs = [q, k_c, v_c, scores] + ([sink_bias] if sink_bias is not None else [])
        _record("aclnn.npu_sparse_attn_sharedkv", inputs, [output], module_path)
        ctx.save_for_backward(q, k_c, v_c, idx_q, idx_k_c)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k_c, v_c, idx_q, idx_k_c = ctx.saved_tensors
        grad_scores = _uncaptured_empty(
            (q.shape[0], q.shape[1], k_c.shape[1]), q.dtype, q.device
        )
        grad_q = _uncaptured_empty_like(q)
        grad_k_c = _uncaptured_empty_like(k_c)
        grad_v_c = _uncaptured_empty_like(v_c)
        grad_idx_q = _uncaptured_empty_like(idx_q)
        grad_idx_k_c = _uncaptured_empty_like(idx_k_c)
        _record(
            "aclnn.npu_lightning_indexer_grad",
            [grad_scores, idx_q, idx_k_c],
            [grad_idx_q, grad_idx_k_c],
            ctx.module_path,
        )
        _record(
            "aclnn.npu_sparse_attn_sharedkv_grad",
            [grad_output, q, k_c, v_c, grad_scores],
            [grad_q, grad_k_c, grad_v_c, grad_scores],
            ctx.module_path,
        )
        return grad_q, grad_k_c, grad_v_c, grad_idx_q, grad_idx_k_c, None, None


def _sim_hca_forward(module, q, k, v, hidden_states, sink_bias):  # noqa: ANN001
    idx_q = module.indexer_q_norm(module.indexer_q(hidden_states))
    idx_k = module.indexer_k_norm(module.indexer_k(hidden_states))
    k_c = k[:, :: module.compress_ratio]
    v_c = v[:, :: module.compress_ratio]
    idx_k_c = idx_k[:, :: module.compress_ratio]
    return _SimHCACoreFn.apply(
        q, k_c, v_c, idx_q, idx_k_c, sink_bias, _current_module_path()
    )


class _SimSinkhornFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mix_weights, n_iters, module_path):  # noqa: ANN001
        del n_iters
        mixed = _uncaptured_empty_like(mix_weights)
        _record(
            "triton._triton_hc_sinkhorn_fwd_kernel",
            [mix_weights],
            [mixed],
            module_path,
        )
        ctx.module_path = module_path
        return mixed

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        grad_weights = _uncaptured_empty_like(grad_output)
        _record(
            "triton._triton_hc_sinkhorn_bwd_kernel",
            [grad_output],
            [grad_weights],
            ctx.module_path,
        )
        return grad_weights, None, None


def _sim_hc_block_forward(module, x):  # noqa: ANN001
    b, s, d = x.shape
    residual = x
    xn = module.pre_norm(x)
    expanded = module.expand(xn).view(b, s, module.hc_mult, d)
    mix = _SimSinkhornFn.apply(
        module.mix_weights, module.sinkhorn.n_iters, _current_module_path()
    )
    mixed = torch.einsum("bshd,ho->bsod", expanded, mix)
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
