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
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config import derive, override
from torchtitan.models.common.attention import LinearAttentionMetadata
from torchtitan.models.common.attention import gdn as upstream_gdn
from torchtitan.models.common.attention.gdn import GDN, GDNKernel

from torchtitan_npu.ops.triton.gdn import gated_delta_rule as run_gdn
from torchtitan_npu.override.qwen3_5.parallelize import (
    QWEN_CONTEXT_PARALLEL_UNSUPPORTED,
)


class TritonGatedDeltaKernel(GDNKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GDNKernel.Config):
        pass

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
            reset = torch.zeros(xq_THK.shape[0], dtype=torch.bool, device=xq_THK.device)
            reset[cu_seqlens[:-1].to(device=reset.device, dtype=torch.long)] = True
            reset = reset.unsqueeze(0)
        return run_gdn(
            xq_THK.unsqueeze(0),
            xk_THK.unsqueeze(0),
            xv_THV.unsqueeze(0),
            g_TH.unsqueeze(0),
            beta_TH.unsqueeze(0),
            reset=reset,
        ).squeeze(0)


@override(target=GDNKernel.Config, exact=True, description="Use the Triton-Ascend GDN kernel")
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


def _causal_conv1d_varlen_folded(
    x_TD: torch.Tensor,
    weight_C1W: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """The folded ``[T, C]`` signature the 0.4.0 ``InnerGDN`` implementation calls.

    Device-side offsets keep the convolution traceable inside checkpoint HOP; the
    eager strict-validation branch of :func:`_causal_conv1d_varlen` cannot run
    because the folded metadata no longer carries CPU-side offsets.
    """
    return _causal_conv1d_varlen_tensor(x_TD.unsqueeze(0), weight_C1W, cu_seqlens).squeeze(0)


@override(target=GDNKernel.Config, exact=True, description="Use the NPU GDN kernel without FLA")
def npu(cfg: GDNKernel.Config) -> TritonGatedDeltaKernel.Config:
    # The upstream 0.4.0 ``InnerGDN`` implementation resolves its packed
    # convolution through the ``gdn`` module global at call time, so swapping
    # that global is the seam the folded upstream actually calls.  FLA's
    # ``causal_conv1d`` is CUDA-only, which is why the reference cannot run
    # as-is on NPU.  FLA availability is not used as a backend selector
    # because its convolution is not an NPU implementation.
    upstream_gdn._causal_conv1d_varlen = _causal_conv1d_varlen_folded  # pyrefly: ignore [bad-assignment]
    return derive(cfg, TritonGatedDeltaKernel.Config)


class ContextParallelGatedDeltaNet(GDN):
    """Refuse the not-yet-migrated context-parallel GDN forward.

    The class preserves the configured CP implementation while its forward
    refuses execution: the pre-fold implementation exchanged rank-local
    heads, convolved ``[B, L, D]`` rows, and called the kernel through
    ``self.kernel``, none of which exists on the folded ``[T, ...]`` module.
    """

    context_parallel_mesh: DeviceMesh

    @dataclass(kw_only=True, slots=True)
    class Config(GDN.Config):
        pass

    def forward(self, x_TD: torch.Tensor, attention_metadata: LinearAttentionMetadata | None = None) -> torch.Tensor:
        raise NotImplementedError(QWEN_CONTEXT_PARALLEL_UNSUPPORTED)


@override(target=GDN.Config, exact=True, description="Use Triton GDN with Context Parallel")
def context_parallel(cfg: GDN.Config) -> ContextParallelGatedDeltaNet.Config:
    return derive(cfg, ContextParallelGatedDeltaNet.Config)
