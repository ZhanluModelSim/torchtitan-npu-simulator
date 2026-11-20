# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=huawei-invalid-name

from collections.abc import Callable
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torchtitan.config import derive, override
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.attention import VarlenAttentionMetadata
from torchtitan.models.common.attention.cp_attention import CPInnerAttention, UlyssesVarlenInnerAttention


class AscVarlenInnerAttention(VarlenInnerAttention):
    """Use the CANN TND operator at the native varlen attention boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenInnerAttention.Config):
        pass

    def __init__(self, config: Config):
        Module.__init__(self)
        if config.window_size != (-1, 0):
            raise ValueError("The Qwen CANN adapter requires full causal attention.")
        self._causal_mask = None

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata,
        scale: float | None = None,
        out_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            raise NotImplementedError(f"Unsupported Qwen CANN attention options: {sorted(kwargs)}")
        if out_transform is not None:
            raise NotImplementedError("NPU CP attention does not provide the LSE required by out_transform.")
        if enable_gqa and q_THK.shape[1] != k_THK.shape[1]:
            repeats = q_THK.shape[1] // k_THK.shape[1]
            k_THK = k_THK.repeat_interleave(repeats, 1)
            v_THV = v_THV.repeat_interleave(repeats, 1)
        cu_q = attention_metadata.cu_seq_q.to(device="cpu", dtype=torch.int64)
        cu_k = attention_metadata.cu_seq_k.to(device="cpu", dtype=torch.int64)
        actual_seq_q = cu_q[1:][cu_q[1:] > cu_q[:-1]].tolist()
        actual_seq_kv = cu_k[1:][cu_k[1:] > cu_k[:-1]].tolist()
        if self._causal_mask is None or self._causal_mask.device != q_THK.device:
            self._causal_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=q_THK.device), diagonal=1)
        output_THV = torch.ops.npu.npu_fusion_attention(
            query=q_THK,
            key=k_THK,
            value=v_THV,
            head_num=q_THK.shape[1],
            input_layout="TND",
            atten_mask=self._causal_mask,
            scale=q_THK.shape[-1] ** -0.5 if scale is None else scale,
            keep_prob=1.0,
            pre_tockens=q_THK.shape[0],
            next_tockens=0,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            sparse_mode=7,
        )[0]
        return output_THV


class AscCPVarlenInnerAttention(
    CPInnerAttention[VarlenAttentionMetadata, VarlenAttentionMetadata], AscVarlenInnerAttention
):
    """Replicate Qwen's four KV heads before Ulysses sharding across eight CP ranks."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, AscVarlenInnerAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        if permutation is not None:
            raise ValueError("NPU head exchange requires contiguous CP sharding without a load balancer.")
        return attention_metadata

    def forward(self, q_THK, k_THK, v_THV, *, out_transform=None, enable_gqa=False, **kwargs):
        if out_transform is not None:
            raise NotImplementedError("NPU CP attention does not provide the LSE required by out_transform.")
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            raise RuntimeError("NPU context-parallel attention requires an active CP group.")
        if q_THK.shape[1] % group.size():
            raise ValueError("Qwen query heads must be divisible by CP.")
        if enable_gqa and k_THK.shape[1] % group.size():
            repeats = q_THK.shape[1] // k_THK.shape[1]
            k_THK = k_THK.repeat_interleave(repeats, 1)
            v_THV = v_THV.repeat_interleave(repeats, 1)
        if k_THK.shape[1] % group.size() or v_THV.shape[1] % group.size():
            raise ValueError("Qwen K/V heads must divide CP, or enable_gqa must allow head replication.")
        seq, heads = spmd.S(0), spmd.S(1)
        q_THK, k_THK, v_THV = (spmd.redistribute(x, group, src=seq, dst=heads) for x in (q_THK, k_THK, v_THV))
        output = super().forward(q_THK, k_THK, v_THV, enable_gqa=enable_gqa, **kwargs)
        return spmd.redistribute(output, group, src=heads, dst=seq)


@override(
    target=UlyssesVarlenInnerAttention.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use CANN TND attention with Qwen CP head exchange",
)
def asc_cp(cfg: UlyssesVarlenInnerAttention.Config) -> AscCPVarlenInnerAttention.Config:
    return derive(cfg, AscCPVarlenInnerAttention.Config)


@override(
    target=VarlenInnerAttention.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use CANN TND attention for Qwen",
)
def asc(cfg: VarlenInnerAttention.Config) -> AscVarlenInnerAttention.Config:
    return derive(cfg, AscVarlenInnerAttention.Config)
