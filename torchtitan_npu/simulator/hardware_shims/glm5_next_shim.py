# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only simulator shims for glm5_next model-specific fused ops.

Records the real production op names (MODEL_CONTRACT.md section 11) into the
active OpDispatchCapture with analytically-correct shapes:

- KDA fused qkv conv:    ``triton_ascend_kernels.causal_conv1d[_grad]``
- KDA core:              ``triton_ascend_kernels.chunk_kda[_grad]``
- DSA indexer selection: ``aclnn.npu_lightning_indexer[_grad]``
- DSA sparse attention:  ``aclnn.npu_sparse_attn_sharedkv[_grad]``

Shims are bound after model construction and parallelization (same seam as
``apply_ar_llm_shims``) so FQN, DTensor placements, and hooks stay intact.
mHC pre/post are covered by the shared DSv4 ``npu_mhc_pre/post`` converters
plus ``apply_mhc_shims`` and need no extra binding here.
"""

from __future__ import annotations

from types import MethodType

import torch

from torchtitan_npu.models.glm5_next.attention import GlmDeltaAttention, GlmDsaAttention, ShortConv1d
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
from torchtitan_npu.simulator.hardware_shims.kda_shim import (
    _current_module_path,
    _record,
    _uncaptured_empty_like,
)


def _uncaptured_empty(shape, dtype, device):  # noqa: ANN001
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=dtype, device=device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=dtype, device=device)

_KDA_SHIM_MARKER = "_simulator_glm5_next_kda_shim_installed"
_CONV_SHIM_MARKER = "_simulator_glm5_next_conv_shim_installed"
_DSA_SHIM_MARKER = "_simulator_glm5_next_dsa_shim_installed"


class _SimChunkKDAFn(torch.autograd.Function):
    """Shape-only bridge for the KDA chunk kernel on meta tensors.

    glm5_next pre-computes the forget gate ``g`` (decay/A_log/lower bound live
    in the gate module), so the kernel receives g directly.
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
            [grad_output],
            grads,
            ctx.module_path,
        )
        return (*grads, None)


def _sim_chunk_kda(module, q, k, v, g, beta):  # noqa: ANN001
    return _SimChunkKDAFn.apply(q, k, v, g, beta, _current_module_path())


class _SimCausalConv1dFn(torch.autograd.Function):
    """Shape-only bridge for the fused depthwise qkv short conv."""

    @staticmethod
    def forward(ctx, x, weight, module_path):  # noqa: ANN001
        output = _uncaptured_empty_like(x)
        _record(
            "triton_ascend_kernels.causal_conv1d",
            [x, weight],
            [output],
            module_path,
        )
        ctx.save_for_backward(x, weight)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        x, weight = ctx.saved_tensors
        grads = [_uncaptured_empty_like(t) for t in (x, weight)]
        _record(
            "triton_ascend_kernels.causal_conv1d_grad",
            [grad_output],
            grads,
            ctx.module_path,
        )
        return (*grads, None)


def _sim_short_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
    weight = self.conv.weight
    if isinstance(weight, torch.distributed.tensor.DTensor):
        weight = weight.to_local()
    if self._local_channels is not None:
        lc = self._local_channels
        weight = torch.cat([weight[s : s + lc] for s in self._channel_starts], dim=0)
    return _SimCausalConv1dFn.apply(x, weight, _current_module_path())


class _SimDsaIndexerFn(torch.autograd.Function):
    """k-pool compressed indexer scoring + top-k selection as one fused op.

    The output width is static (``select_pools * kpool + tail``); the indexer
    is frozen so backward records are dependency-graph placeholders only.
    """

    @staticmethod
    def forward(ctx, hidden_states, q_resid, output_width, module_path):  # noqa: ANN001
        b, s = hidden_states.shape[0], hidden_states.shape[1]
        topk_indices = _uncaptured_empty((b, s, output_width), torch.int64, hidden_states.device)
        _record(
            "aclnn.npu_lightning_indexer",
            [hidden_states, q_resid],
            [topk_indices],
            module_path,
        )
        ctx.save_for_backward(hidden_states, q_resid)
        ctx.module_path = module_path
        return topk_indices

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        hidden_states, q_resid = ctx.saved_tensors
        grads = [_uncaptured_empty_like(t) for t in (hidden_states, q_resid)]
        _record(
            "aclnn.npu_lightning_indexer_grad",
            [grad_output],
            grads,
            ctx.module_path,
        )
        return *grads, None, None


class _SimDsaCoreFn(torch.autograd.Function):
    """Sparse attention over the top-k union as one fused op."""

    @staticmethod
    def forward(ctx, q, k, v, topk_indices, module_path):  # noqa: ANN001
        output = _uncaptured_empty_like(q)
        _record(
            "aclnn.npu_sparse_attn_sharedkv",
            [q, k, v, topk_indices],
            [output],
            module_path,
        )
        ctx.save_for_backward(q, k, v, topk_indices)
        ctx.module_path = module_path
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        q, k, v, topk_indices = ctx.saved_tensors
        grad_q = _uncaptured_empty_like(q)
        grad_k = _uncaptured_empty_like(k)
        grad_v = _uncaptured_empty_like(v)
        _record(
            "aclnn.npu_sparse_attn_sharedkv_grad",
            [grad_output, q, k, v, topk_indices],
            [grad_q, grad_k, grad_v],
            ctx.module_path,
        )
        return grad_q, grad_k, grad_v, None, None


def _sim_dsa_forward(self, hidden_states, attention_masks=None, positions=None):  # noqa: ANN001
    del attention_masks, positions
    b, s, _ = hidden_states.shape

    q_resid = self.q_a_norm(self.q_a_proj(hidden_states))
    query = self.q_b_proj(q_resid).view(b, s, self.num_heads, self.qk_nope_head_dim)

    kv_pass = self.kv_a_norm(self.kv_a_proj_with_mqa(hidden_states))
    kv = self.kv_b_proj(kv_pass).view(b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    key, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    with torch.no_grad():
        indexer = self.indexer
        select_pools = min(indexer.topk // indexer.kpool, s // indexer.kpool)
        output_width = select_pools * indexer.kpool
        if indexer.always_select_tail:
            output_width += indexer.kpool - 1
        topk_indices = _SimDsaIndexerFn.apply(
            hidden_states, q_resid, output_width, _current_module_path()
        )
    output = _SimDsaCoreFn.apply(query, key, value, topk_indices, _current_module_path())
    return self.o_proj(output.reshape(b, s, self.num_heads * self.v_head_dim))


def apply_glm5_next_shims(model) -> None:
    """Bind glm5_next shape-only shims while preserving module hooks."""
    for module in model.modules():
        if isinstance(module, GlmDeltaAttention):
            if not getattr(module, _KDA_SHIM_MARKER, False):
                module._chunk_kda = MethodType(_sim_chunk_kda, module)
                setattr(module, _KDA_SHIM_MARKER, True)
        elif isinstance(module, ShortConv1d):
            if not getattr(module, _CONV_SHIM_MARKER, False):
                module.forward = MethodType(_sim_short_conv_forward, module)
                setattr(module, _CONV_SHIM_MARKER, True)
        elif isinstance(module, GlmDsaAttention):
            if not getattr(module, _DSA_SHIM_MARKER, False):
                module.forward = MethodType(_sim_dsa_forward, module)
                setattr(module, _DSA_SHIM_MARKER, True)
