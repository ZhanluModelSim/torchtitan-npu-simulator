# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# pylint: disable=huawei-invalid-name

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config import derive, override
from torchtitan.models import qwen3_5
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.qwen3_5.model import GatedDeltaKernel

from torchtitan_npu.ops.triton.gdn import gated_delta_rule as run_gdn
from torchtitan_npu.override.qwen3_5.parallelize import (
    QwenCPMetadata,
    exchange_sequence_heads,
    head_to_sequence_shard,
    shard_local_heads,
)


class TritonGatedDeltaKernel(GatedDeltaKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        pass

    def forward(
        self,
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reset = None
        if cu_seqlens is not None:
            reset = torch.zeros(xq_BLNK.shape[:2], dtype=torch.bool, device=xq_BLNK.device)
            reset.view(-1)[cu_seqlens[:-1].to(device=reset.device, dtype=torch.long)] = True
        return run_gdn(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, reset=reset)


@override(target=GatedDeltaKernel.Config, exact=True, description="Use the Triton-Ascend GDN kernel")
def triton(cfg: GatedDeltaKernel.Config) -> TritonGatedDeltaKernel.Config:
    return derive(cfg, TritonGatedDeltaKernel.Config)


class ContextParallelGatedDeltaNet(qwen3_5.GatedDeltaNet):
    context_parallel_mesh: DeviceMesh

    @dataclass(kw_only=True, slots=True)
    class Config(qwen3_5.GatedDeltaNet.Config):
        pass

    def causal_convolution(self, tensor, convolution, metadata):
        mesh = self.context_parallel_mesh
        weight = shard_local_heads(convolution.weight, mesh)[:, 0]
        reset = torch.zeros(tensor.shape[:2], dtype=torch.bool, device=tensor.device)
        reset.view(-1)[metadata.cu_seqlens[1:-1].to(reset.device)] = True
        segments = reset.cumsum(1)
        output = tensor * weight[:, -1]
        for delay in range(1, self.conv_kernel_size):
            source = tensor[:, :-delay] * weight[:, -1 - delay]
            output[:, delay:] += source * (segments[:, delay:] == segments[:, :-delay]).unsqueeze(-1)
        return F.silu(output)

    def forward(self, x_BLD: torch.Tensor, attention_masks: AttentionMasksType | None = None):
        batch, local_length, _ = x_BLD.shape
        mesh = self.context_parallel_mesh
        metadata = cast("QwenCPMetadata", attention_masks)
        query, key, value, decay, beta = exchange_sequence_heads(
            tuple(
                projection(x_BLD)
                for projection in (self.in_proj_q, self.in_proj_k, self.in_proj_v, self.in_proj_a, self.in_proj_b)
            ),
            mesh,
            2,
        )
        output_gate = self.in_proj_z(x_BLD).view(batch, local_length, -1, self.value_head_dim)
        length = query.size(1)
        query = self.causal_convolution(query, self.conv_q, metadata).view(batch, length, -1, self.key_head_dim)
        key = self.causal_convolution(key, self.conv_k, metadata).view(batch, length, -1, self.key_head_dim)
        value = self.causal_convolution(value, self.conv_v, metadata).view(batch, length, -1, self.value_head_dim)
        a_log = shard_local_heads(self.A_log, mesh)
        dt_bias = shard_local_heads(self.dt_bias, mesh)
        gate = -a_log.float().exp() * F.softplus(decay.float() + dt_bias)
        output = self.kernel(query, key, value, gate, beta.sigmoid(), cu_seqlens=metadata.cu_seqlens)
        output = head_to_sequence_shard(output, mesh, 2)
        return self.out_proj(self.norm(output, output_gate).flatten(2))


@override(target=qwen3_5.GatedDeltaNet.Config, exact=True, description="Use Triton GDN with Context Parallel")
def context_parallel(cfg: qwen3_5.GatedDeltaNet.Config) -> ContextParallelGatedDeltaNet.Config:
    return derive(cfg, ContextParallelGatedDeltaNet.Config, kernel=derive(cfg.kernel, TritonGatedDeltaKernel.Config))
