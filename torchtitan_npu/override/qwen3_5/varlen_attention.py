# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=huawei-invalid-name

from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttention

from torchtitan_npu.override.qwen3_5.parallelize import (
    exchange_sequence_heads,
    head_to_sequence_shard,
)

_CAUSAL_MASK = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)


class AscVarlenAttention(VarlenAttention):
    context_parallel_mesh: DeviceMesh

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        pass

    def forward(self, q_BLNH, k_BLNH, v_BLNH, *, attention_masks, scale=None, enable_gqa=False, **kwargs):
        del kwargs
        mesh = self.context_parallel_mesh
        if enable_gqa and k_BLNH.size(2) % mesh.size():
            repeats = q_BLNH.size(2) // k_BLNH.size(2)
            k_BLNH, v_BLNH = (tensor.repeat_interleave(repeats, 2) for tensor in (k_BLNH, v_BLNH))
        actual_seq = attention_masks.actual_seq_qlen
        q_BLNH, k_BLNH, v_BLNH = exchange_sequence_heads((q_BLNH, k_BLNH, v_BLNH), mesh, 2)
        batch, length, heads, head_dim = q_BLNH.shape
        dtype = q_BLNH.dtype
        query, key, value = (
            tensor.reshape(batch * length, tensor.size(2), head_dim).to(torch.bfloat16)
            for tensor in (q_BLNH, k_BLNH, v_BLNH)
        )
        causal_mask = getattr(self, "_causal_mask", None)
        if causal_mask is None or causal_mask.device != query.device:
            self._causal_mask = causal_mask = _CAUSAL_MASK.to(query.device)
        output = (
            torch.ops.npu.npu_fusion_attention_v3(
                query=query,
                key=key,
                value=value,
                head_num=heads,
                input_layout="TND",
                atten_mask=causal_mask,
                scale=head_dim**-0.5 if scale is None else scale,
                keep_prob=1.0,
                pre_tockens=length,
                next_tockens=0,
                actual_seq_qlen=actual_seq,
                actual_seq_kvlen=actual_seq,
                sparse_mode=7,
            )[0]
            .view(batch, length, heads, head_dim)
            .to(dtype)
        )
        return head_to_sequence_shard(output, mesh, 2)


@override(
    target=VarlenAttention.Config,
    fqns=["model_spec.model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use CANN TND attention with Qwen3.5/3.6 Context Parallel",
)
def asc_cp(cfg: VarlenAttention.Config) -> AscVarlenAttention.Config:
    return derive(cfg, AscVarlenAttention.Config)
