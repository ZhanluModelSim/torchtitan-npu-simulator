# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# pylint: disable=huawei-invalid-name

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F
from torchtitan.config import derive, override
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import LinearAttentionMetadata
from torchtitan.models.common.attention.cp_attention import CPInnerAttention
from torchtitan.models.common.attention.gdn import GDNKernel, InnerGDN
from torchtitan.models.common.norm import GatedRMSNorm

from torchtitan_npu.ops.triton.gdn import gated_delta_rule as run_gdn

from .parallelize import exchange_sequence_heads, head_to_sequence_shard, shard_local_heads


class TritonGatedDeltaKernel(GDNKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GDNKernel.Config):
        pass

    def __init__(self, config: Config):
        # The native CUDA constructor checks for Attention Gym kernels.
        Module.__init__(self)

    def forward(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # One reset flag per row of the packed row; ``cu_seqlens[:-1]`` are the
        # document starts.  The Triton-Ascend op speaks the batched ``[B, T, H, K]``
        # layout, so the folded caller's single row goes on as ``B=1`` and the
        # result's batch axis comes back off.
        reset = None
        if cu_seqlens is not None:
            # Fixed-capacity upstream metadata repeats the terminal offset.
            # Keep a sentinel slot so these empty segments never index past T.
            reset = torch.zeros(xq_THK.shape[0] + 1, dtype=torch.bool, device=xq_THK.device)
            reset[cu_seqlens[:-1].to(device=reset.device, dtype=torch.long)] = True
            reset = reset[:-1].unsqueeze(0)
        return run_gdn(
            xq_THK.unsqueeze(0),
            xk_THK.unsqueeze(0),
            xv_THV.unsqueeze(0),
            g_TH.unsqueeze(0),
            beta_TH.unsqueeze(0),
            reset=reset,
        ).squeeze(0)


@override(
    target=GDNKernel.Config,
    exact=True,
    description="Use the Triton-Ascend GDN kernel",
)
def triton(cfg: GDNKernel.Config) -> TritonGatedDeltaKernel.Config:
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
    boundary_mask = torch.zeros(x_BTD.shape[1] + 1, dtype=torch.bool, device=x_BTD.device)
    boundary_mask = boundary_mask.scatter(
        0,
        boundary_indices,
        torch.ones_like(boundary_indices, dtype=torch.bool),
    )
    segment_ids = boundary_mask[:-1].cumsum(dim=0)

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


def _causal_conv1d_varlen_folded(
    x_TD: torch.Tensor,
    weight_C1W: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Apply the packed tensor convolution at the native InnerGDN hook."""
    return _causal_conv1d_varlen_tensor(x_TD.unsqueeze(0), weight_C1W, cu_seqlens).squeeze(0)


class AscInnerGDN(InnerGDN):
    """NPU convolution and the existing repository GDN custom op.

    This adapter owns the packed-convolution policy. The Triton forward,
    backward and autograd registration are consumed unchanged from ops/.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerGDN.Config):
        pass

    def __init__(self, config: Config):
        Module.__init__(self)
        self.kernel = config.kernel.build()

    def short_convolution(self, x_TC, weight_C1W, initial_state=None, *, cu_seqlens, routing):
        if initial_state is not None or routing is not None:
            raise ValueError("NPU head exchange consumes complete sequences without recurrent routing.")
        if cu_seqlens is None or cu_seqlens.numel() == 2:
            return _causal_conv1d(x_TC.unsqueeze(0), weight_C1W).squeeze(0)
        return _causal_conv1d_varlen_folded(x_TC, weight_C1W, cu_seqlens)


class AscContextParallelInnerGDN(CPInnerAttention[LinearAttentionMetadata, LinearAttentionMetadata], AscInnerGDN):
    """CP8 exchanges packed GDN inputs before causal convolution so document resets stay global."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, AscInnerGDN.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        if permutation is not None:
            raise ValueError("NPU GDN head exchange requires contiguous CP sharding without a load balancer.")
        return attention_metadata

    def forward(
        self,
        query_TC,
        key_TC,
        value_TC,
        a_TH,
        b_TH,
        conv_q_weight_C1W,
        conv_k_weight_C1W,
        conv_v_weight_C1W,
        A_log_H,
        dt_bias_H,
        *,
        key_head_dim,
        value_head_dim,
        attention_metadata: LinearAttentionMetadata | None,
    ):
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            raise RuntimeError("NPU context-parallel GDN requires an active CP group.")
        if any(
            tensor.shape[1] % (group.size() * dimension)
            for tensor, dimension in ((query_TC, key_head_dim), (key_TC, key_head_dim), (value_TC, value_head_dim))
        ):
            raise ValueError("GDN query, key and value head counts must be divisible by CP.")
        query_TC, key_TC, value_TC, a_TH, b_TH = exchange_sequence_heads(
            (query_TC, key_TC, value_TC, a_TH, b_TH), group
        )
        parameters = tuple(
            shard_local_heads(tensor, group)
            for tensor in (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W, A_log_H, dt_bias_H)
        )
        conv_q, conv_k, conv_v, A_log, dt_bias = parameters
        output = super().forward(
            query_TC,
            key_TC,
            value_TC,
            a_TH,
            b_TH,
            conv_q,
            conv_k,
            conv_v,
            A_log,
            dt_bias,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            attention_metadata=attention_metadata,
        )
        return head_to_sequence_shard(output, group)


@override(
    target=InnerGDN.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use NPU convolution and the built-in Triton GDN",
)
def npu(cfg: InnerGDN.Config) -> AscInnerGDN.Config:
    return derive(cfg, AscInnerGDN.Config, kernel=derive(cfg.kernel, TritonGatedDeltaKernel.Config))


@override(
    target=AscContextParallelInnerGDN.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use NPU GDN with full-sequence head exchange",
)
def context_parallel(cfg: AscContextParallelInnerGDN.Config) -> AscContextParallelInnerGDN.Config:
    return derive(cfg, AscContextParallelInnerGDN.Config, kernel=derive(cfg.kernel, TritonGatedDeltaKernel.Config))


class QwenSFTGatedRMSNorm(GatedRMSNorm):
    """Preserve the Qwen SFT rounding point before multiplying by the gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(GatedRMSNorm.Config):
        pass

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.eps)
        normalized = (self.weight.float() * values).to(hidden_states.dtype)
        return (normalized * self.activation_fn(gate.float())).to(hidden_states.dtype)


@override(
    target=GatedRMSNorm.Config,
    fqns=["model.layers.*.attention.norm"],
    exact=True,
    description="Preserve the Qwen SFT normalization and gating precision",
)
def qwen_sft(cfg: GatedRMSNorm.Config) -> QwenSFTGatedRMSNorm.Config:
    return derive(cfg, QwenSFTGatedRMSNorm.Config)
