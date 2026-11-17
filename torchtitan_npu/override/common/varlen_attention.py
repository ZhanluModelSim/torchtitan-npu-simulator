# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: dense GQA varlen attention backed by CANN ``npu_fusion_attention_v3``.

This is the dense-decoder counterpart of the qwen3.5 CP attention path: the
packed-document TND call with ``actual_seq_qlen`` varlen semantics and a
shared causal mask, but without the context-parallel head exchange.  The
kernel boundary is the real NPU training op, so the simulator captures
``npu::npu_fusion_attention_v3`` forward and
``npu::npu_fusion_attention_grad_v3`` backward nodes instead of a decomposed
or synthetic attention graph.  On Meta the simulator side installs an
autograd bridge through the module-level ``_fusion_attention_v3`` seam
(see ``torchtitan_simulator.compat.attention``).
"""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata

__all__ = ["AscVarlenAttention", "asc"]


def _fusion_attention_v3(
    *,
    query,
    key,
    value,
    atten_mask,
    actual_seq,
    head_num,
    scale,
    pre_tockens,
    next_tockens,
):
    """Single seam for the fused op; the simulator replaces it on Meta devices."""
    result = torch_npu.npu_fusion_attention_v3(
        query=query,
        key=key,
        value=value,
        head_num=head_num,
        input_layout="TND",
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=1.0,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        actual_seq_qlen=actual_seq,
        actual_seq_kvlen=actual_seq,
        sparse_mode=7,
    )
    return result[0]


class AscVarlenAttention(VarlenAttention):
    """Dense causal GQA attention on the fused CANN TND kernel.

    The op natively supports grouped queries: ``head_num`` is the query head
    count and key/value keep their own head count.  Casts mirror the native
    VarlenAttention contract (the kernel consumes bf16 inputs); the causal
    mask is cached per module instance so a warmed-up capture reuses it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        pass

    def forward(self, q_BLNH, k_BLNH, v_BLNH, *, attention_masks, scale=None, enable_gqa=False, **kwargs):
        del kwargs, enable_gqa  # GQA is expressed by the head counts themselves.
        if self.window_size != (-1, 0):
            raise NotImplementedError(
                f"AscVarlenAttention supports full causal attention only, got window_size={self.window_size}"
            )
        if not isinstance(attention_masks, VarlenMetadata):
            raise TypeError(f"AscVarlenAttention requires VarlenMetadata, got {type(attention_masks).__name__}")
        batch, length, hq, head_dim = q_BLNH.shape
        hkv = k_BLNH.shape[2]
        if v_BLNH.shape[-1] != head_dim:
            raise NotImplementedError("AscVarlenAttention requires equal key/value head dimensions")
        total = batch * length
        # TND packing matches the native VarlenAttention reshape contract.
        query = q_BLNH.reshape(total, hq, head_dim).to(torch.bfloat16)
        key = k_BLNH.reshape(total, hkv, head_dim).to(torch.bfloat16)
        value = v_BLNH.reshape(total, hkv, head_dim).to(torch.bfloat16)
        # Ascend varlen convention: cumulative document lengths WITHOUT the
        # leading zero (cu_seq_q carries [0, s1, s1+s2, ...]).
        actual_seq = attention_masks.cu_seq_q[1:].to(torch.int64)
        # sparse_mode=7 consumes a fixed compressed causal mask, independent
        # of the document lengths carried by actual_seq.
        mask_len = 2048
        causal_mask = getattr(self, "_causal_mask", None)
        if causal_mask is None or causal_mask.device != query.device:
            causal_mask = torch.triu(torch.ones(mask_len, mask_len, dtype=torch.bool, device=query.device), diagonal=1)
            self._causal_mask = causal_mask
        out = _fusion_attention_v3(
            query=query,
            key=key,
            value=value,
            atten_mask=causal_mask,
            actual_seq=actual_seq,
            head_num=hq,
            scale=head_dim**-0.5 if scale is None else scale,
            pre_tockens=length,
            next_tockens=0,
        )
        return out.view(batch, length, hq, head_dim).to(q_BLNH.dtype)


@override(
    target=VarlenAttention.Config,
    fqns=["model_spec.model.layers.*.attention.inner_attention"],
    exact=True,
    description="Dense CANN TND fusion attention via npu_fusion_attention_v3",
)
def asc(cfg: VarlenAttention.Config) -> "AscVarlenAttention.Config":
    return derive(cfg, AscVarlenAttention.Config)
