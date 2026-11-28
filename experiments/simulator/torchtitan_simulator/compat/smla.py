# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Carry host-known DeepSeek V4 sequence lengths across native SMLA autograd."""

from contextvars import ContextVar
from functools import wraps
from itertools import pairwise

from torchtitan_simulator.capture.dispatch_capture import capture_attention_geometry

_forward_geometry = ContextVar("smla_forward_geometry", default=None)


def _lengths(offsets):
    return tuple(end - begin for begin, end in pairwise(offsets))


def install_deepseek_v4_smla_geometry(scope, owner):
    attention = owner.AscCompressedSparseInnerAttention
    bridge = owner._SparseFlashMLATND
    original_attention = attention.forward
    original_forward = bridge.forward
    original_backward = bridge.backward

    @wraps(original_attention)
    def attention_forward(module, *args, **kwargs):
        metadata = kwargs.get("attention_masks")
        if metadata is None:
            return original_attention(module, *args, **kwargs)
        varlen = metadata.varlen
        q = getattr(varlen, "cu_seq_q_host", None)
        if q is None:
            q = getattr(varlen.cu_seq_q, "_sim_host_attention_values", None)
        if q is None:
            return original_attention(module, *args, **kwargs)
        # Without a packed window the autograd bridge uses cu_seqlens_q for original KV.
        ori = q
        if metadata.window is not None:
            ori_offsets = metadata.window.cu_seqlens_ori_kv
            ori = getattr(ori_offsets, "_sim_host_attention_values", None)
            if ori is None:
                return original_attention(module, *args, **kwargs)
        facts = {"query_lens": _lengths(q), "kv_lens": _lengths(ori)}
        plan = metadata.plans[module.compress_ratio]
        if plan.cu_seqlens_cmp_k is not None:
            cmp = getattr(plan.cu_seqlens_cmp_k, "_sim_host_attention_values", None)
            if cmp is None:
                return original_attention(module, *args, **kwargs)
            facts["compressed_lens"] = _lengths(cmp)
        token = _forward_geometry.set(facts)
        try:
            return original_attention(module, *args, **kwargs)
        finally:
            _forward_geometry.reset(token)

    @wraps(original_forward)
    def forward(ctx, *args, **kwargs):
        facts = _forward_geometry.get()
        ctx._sim_smla_geometry = facts
        if facts is None:
            return original_forward(ctx, *args, **kwargs)
        with capture_attention_geometry("cann_ops_transformer.sparse_flash_mla.default", **facts):
            return original_forward(ctx, *args, **kwargs)

    @wraps(original_backward)
    def backward(ctx, *args, **kwargs):
        facts = ctx._sim_smla_geometry
        if facts is None:
            return original_backward(ctx, *args, **kwargs)
        with capture_attention_geometry("cann_ops_transformer.sparse_flash_mla_grad.default", **facts):
            return original_backward(ctx, *args, **kwargs)

    scope.replace(attention, "forward", attention_forward)
    scope.replace(bridge, "forward", staticmethod(forward))
    scope.replace(bridge, "backward", staticmethod(backward))
