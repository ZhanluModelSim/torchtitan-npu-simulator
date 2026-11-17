# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# pylint: disable=huawei-invalid-name

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override
from torchtitan.models.common.attention import AttentionMasksType

from torchtitan_npu.models.qwen3_5._fla_compat import ensure_qwen3_5_importable

ensure_qwen3_5_importable()

from torchtitan.models import qwen3_5
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


def _causal_conv1d(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Depthwise causal convolution used when FLA is not installed."""
    if weight.ndim == 3:
        weight = weight.squeeze(1)
    if weight.ndim != 2:
        raise ValueError(f"Expected depthwise conv weight [channels, kernel], got {weight.shape}")
    dilation_value = dilation[0] if isinstance(dilation, tuple) else dilation
    x_BDL = F.pad(x_BTD.transpose(1, 2), ((weight.shape[-1] - 1) * dilation_value, 0))
    out_BDL = F.conv1d(
        x_BDL,
        weight.unsqueeze(1),
        None,
        stride,
        padding,
        dilation,
        weight.size(0),
    )
    return F.silu(out_BDL).transpose(1, 2)


_QWEN3_5_CONV_KERNEL_SIZE = 4


def _causal_conv1d_varlen_tensor(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Compile-safe packed causal convolution using device-side offsets."""
    if weight.ndim == 3:
        weight = weight.squeeze(1)
    if weight.ndim != 2:
        raise ValueError(f"Expected depthwise conv weight [channels, kernel], got {weight.shape}")
    stride_value = stride[0] if isinstance(stride, tuple) else stride
    padding_value = padding[0] if isinstance(padding, tuple) else padding
    dilation_value = dilation[0] if isinstance(dilation, tuple) else dilation
    if stride_value != 1 or padding_value != 0 or dilation_value != 1 or weight.shape[-1] != _QWEN3_5_CONV_KERNEL_SIZE:
        raise ValueError("Qwen3.5 compiled varlen convolution requires kernel=4, stride=1, padding=0, dilation=1")

    boundary_indices = cu_seqlens[1:-1].to(device=x_BTD.device, dtype=torch.long)
    boundary_mask = torch.zeros(x_BTD.shape[1], dtype=torch.bool, device=x_BTD.device)
    boundary_mask = boundary_mask.scatter(
        0,
        boundary_indices,
        torch.ones_like(boundary_indices, dtype=torch.bool),
    )
    segment_ids = boundary_mask.cumsum(dim=0)

    output = x_BTD * weight[:, -1].view(1, 1, -1)
    for delay in range(1, _QWEN3_5_CONV_KERNEL_SIZE):
        same_segment = (segment_ids[delay:] == segment_ids[:-delay]).to(x_BTD.dtype).view(1, -1, 1)
        term = x_BTD[:, :-delay, :] * weight[:, -1 - delay].view(1, 1, -1)
        output = output + F.pad(term * same_segment, (0, 0, delay, 0))
    return F.silu(output)


def _causal_conv1d_varlen(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
) -> torch.Tensor:
    """Apply causal convolution independently to each packed document.

    Compiled execution consumes device-side offsets so checkpoint HOP can trace
    the operation; eager execution retains strict host-metadata validation.
    """
    if torch.compiler.is_compiling():
        if cu_seqlens is None:
            raise ValueError("Qwen3.5 compiled varlen convolution requires device cu_seqlens metadata.")
        return _causal_conv1d_varlen_tensor(
            x_BTD,
            weight,
            cu_seqlens,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    if cu_seqlens_cpu is None:
        raise ValueError("Qwen3.5 varlen causal convolution requires CPU cu_seqlens metadata.")
    offsets = [int(offset) for offset in cu_seqlens_cpu.detach().cpu().tolist()]
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != x_BTD.shape[1]:
        raise ValueError(f"Invalid Qwen3.5 cu_seqlens metadata: offsets={offsets}, sequence_length={x_BTD.shape[1]}")
    outputs = []
    for start, end in pairwise(offsets):
        if start < 0 or end <= start:
            raise ValueError(f"Qwen3.5 cu_seqlens must be monotonic: {offsets}")
        outputs.append(
            _causal_conv1d(
                x_BTD[:, start:end, :],
                weight,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
    return torch.cat(outputs, dim=1) if outputs else x_BTD[:, :0, :]


def _npu_causal_conv(
    self,
    x_BLD: torch.Tensor,
    conv,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use the NPU-safe reference convolution instead of lazy FLA imports."""
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    if cu_seqlens is not None:
        if isinstance(x_BLD, DTensor):

            def _conv_varlen(x_local_BLD, w_local, cu_seqlens_local):
                return _causal_conv1d_varlen(
                    x_local_BLD,
                    w_local,
                    cu_seqlens_cpu,
                    cu_seqlens=cu_seqlens_local,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )

            return self._local_map_conv(x_BLD, conv, _conv_varlen, cu_seqlens)
        return _causal_conv1d_varlen(
            x_BLD,
            conv.weight,
            cu_seqlens_cpu,
            cu_seqlens=cu_seqlens,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    if isinstance(x_BLD, DTensor):

        def _conv_fixed(x_local_BLD, w_local):
            return _causal_conv1d(
                x_local_BLD,
                w_local,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        return self._local_map_conv(x_BLD, conv, _conv_fixed)
    return _causal_conv1d(
        x_BLD,
        conv.weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


@override(target=GatedDeltaKernel.Config, exact=True, description="Use the NPU GDN kernel without FLA")
def npu(cfg: GatedDeltaKernel.Config) -> TritonGatedDeltaKernel.Config:
    # The upstream GatedDeltaNet config remains the owner of the module.  Only
    # its kernel config and convolution implementation are replaced, matching
    # the upstream build contract and keeping pipeline stage metadata unchanged.
    # FLA availability is not used as a backend selector because its
    # convolution is not an NPU implementation.
    qwen3_5.GatedDeltaNet._causal_conv = _npu_causal_conv  # pyrefly: ignore [bad-assignment]
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
