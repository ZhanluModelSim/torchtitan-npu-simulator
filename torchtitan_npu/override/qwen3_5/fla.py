# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional official FLA operators for Qwen's inner GDN."""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import torch
import torch_remat as remat
from torchtitan.config import derive, override
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention.attention import local_head_split
from torchtitan.models.common.attention.gdn import GDN, GDNKernel, InnerGDN
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input

try:
    from fla.modules.causal_conv1d import causal_conv1d  # pyrefly: ignore [missing-import]
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # pyrefly: ignore [missing-import]
except ImportError as error:
    raise ImportError(
        "Install the supported flash-linear-attention and fla-core packages to use the Qwen FLA override."
    ) from error

from .gated_delta import AscContextParallelInnerGDN, AscInnerGDN

if TYPE_CHECKING:
    from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
    from torchtitan.trainer import Trainer


@lru_cache(maxsize=8)
def _packed_offsets(cu_seqlens: torch.Tensor | None):
    # The upstream model creates immutable offsets for each microbatch.
    if cu_seqlens is None:
        return None, None
    offsets_cpu = cu_seqlens.to(device="cpu", dtype=torch.int64).unique_consecutive()
    return offsets_cpu.to(cu_seqlens.device), offsets_cpu


class FLAGatedDeltaKernel(GDNKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(GDNKernel.Config):
        force_fp32: bool = False

    def __init__(self, config: Config):
        # Forward uses FLA directly, so it needs no Attention Gym kernels.
        Module.__init__(self)
        self.force_fp32 = config.force_fp32

    # TorchTitan passes layout-specific tensors positionally.
    def forward(  # pyrefly: ignore [bad-param-name-override]
        self, xq_thk, xk_thk, xv_thv, g_th, beta_th, *, cu_seqlens=None
    ):
        offsets, _ = _packed_offsets(cu_seqlens)
        inputs = (xq_thk, xk_thk, xv_thv, g_th, beta_th)
        if self.force_fp32:
            inputs = tuple(tensor.float() for tensor in inputs)
        gated_delta = cast("Any", chunk_gated_delta_rule)
        output, _ = gated_delta(
            *(tensor.unsqueeze(0) for tensor in inputs),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=offsets,
            chunk_size=64,
        )
        return output.squeeze(0).to(xv_thv.dtype)


def _convolution(x, weight, cu_seqlens, *, force_fp32: bool, preserve_stride: bool):
    dtype = x.dtype
    if force_fp32:
        x, weight = x.float(), weight.float()
    offsets, offsets_cpu = _packed_offsets(cu_seqlens)
    output, _ = cast("Any", causal_conv1d)(
        x=(x if preserve_stride else x.contiguous()).unsqueeze(0),
        weight=weight.squeeze(1),
        bias=None,
        activation="silu",
        backend="triton",
        cu_seqlens=offsets,
        cu_seqlens_cpu=offsets_cpu,
    )
    return output.squeeze(0).to(dtype)


class FLAInnerGDN(AscInnerGDN):
    @dataclass(kw_only=True, slots=True)
    class Config(AscInnerGDN.Config):
        causal_conv1d: bool = True
        force_fp32: bool = False
        preserve_conv_stride: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_fla_convolution = config.causal_conv1d
        self.force_fp32 = config.force_fp32
        self.preserve_conv_stride = config.preserve_conv_stride

    # InnerGDN passes the activation and weight positionally.
    def short_convolution(  # pyrefly: ignore [bad-param-name-override]
        self, x_tc, weight_c1w, initial_state=None, *, cu_seqlens, routing
    ):
        if initial_state is not None or routing is not None or not self.use_fla_convolution:
            return super().short_convolution(x_tc, weight_c1w, initial_state, cu_seqlens=cu_seqlens, routing=routing)
        return _convolution(
            x_tc,
            weight_c1w,
            cu_seqlens,
            force_fp32=self.force_fp32,
            preserve_stride=self.preserve_conv_stride,
        )


class FLAContextParallelInnerGDN(AscContextParallelInnerGDN, FLAInnerGDN):
    """Reuse Qwen's packed CP8 exchange for single-node 64K and two-node 256K."""

    @dataclass(kw_only=True, slots=True)
    class Config(AscContextParallelInnerGDN.Config, FLAInnerGDN.Config):
        pass


@override(
    target=InnerGDN.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    description="Use official optional FLA operators for Qwen",
)
def operators(
    cfg: InnerGDN.Config, *, context_parallel: bool | None = None, causal_conv1d: bool = True, force_fp32: bool = False
) -> FLAInnerGDN.Config | FLAContextParallelInnerGDN.Config:
    uses_cp = isinstance(cfg, AscContextParallelInnerGDN.Config)
    if context_parallel is not None and context_parallel != uses_cp:
        raise ValueError("FLA context_parallel must match the Qwen recipe's GDN layout.")
    kernel = derive(cfg.kernel, FLAGatedDeltaKernel.Config, force_fp32=force_fp32)
    if uses_cp:
        return derive(
            cfg, FLAContextParallelInnerGDN.Config, kernel=kernel, causal_conv1d=causal_conv1d, force_fp32=force_fp32
        )
    return derive(cfg, FLAInnerGDN.Config, kernel=kernel, causal_conv1d=causal_conv1d, force_fp32=force_fp32)


@override(
    target=InnerGDN.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    description="Use official FLA GDN and convolution with Qwen CP head exchange",
)
def context_parallel(cfg: InnerGDN.Config) -> FLAContextParallelInnerGDN.Config:
    if not isinstance(cfg, AscContextParallelInnerGDN.Config):
        raise ValueError("The Qwen FLA override requires a context-parallel GDN recipe.")
    result = operators(cfg, context_parallel=True)
    assert isinstance(result, FLAContextParallelInnerGDN.Config)
    return result


class _MasterOrderGDN(GDN):
    @dataclass(kw_only=True, slots=True)
    class Config(GDN.Config):
        pass

    # Form the projection nodes in master's order for BF16 gradient accumulation.
    def forward(  # pyrefly: ignore [bad-param-name-override]
        self, hidden_states: torch.Tensor, attention_metadata: "LinearAttentionMetadata | None" = None
    ) -> torch.Tensor:
        hidden_states = maybe_gather_tp_input(self, hidden_states)
        projections = (self.in_proj_q, self.in_proj_k, self.in_proj_v, self.in_proj_a, self.in_proj_b, self.in_proj_z)
        query, key, value, decay, beta, gate = (projection(hidden_states) for projection in projections)
        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            query,
            key,
            value,
            decay,
            beta,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            attention_metadata=attention_metadata,
        )
        gate = local_head_split(gate, self.value_head_dim, cp_shard_dim=0)
        remat.recompute_needs_tensor(output, gate)
        output = self.norm(output, gate)
        return self.out_proj(output.reshape(hidden_states.shape[0], -1))


def use_master_fla(config: "Trainer.Config") -> None:
    """Select master's projection, parameter and FLA input layouts for packed Qwen SFT."""
    from torchtitan_npu.models.qwen3_5 import Qwen35TextModel, split_gate_up

    if not isinstance(config.model, Qwen35TextModel.Config) or config.parallelism.context_parallel_degree != 8:
        raise ValueError("The Qwen master FLA layout requires a dense text recipe with CP8.")
    split_gate_up(config.model, enable_sequence_parallel=config.parallelism.enable_sequence_parallel)
    for layer in config.model.layers:
        attention = layer.attention
        if isinstance(attention, GDN.Config):
            if not isinstance(attention.inner_attention, AscContextParallelInnerGDN.Config):
                raise TypeError("The Qwen master FLA layout requires the native CP GDN configuration.")
            inner = derive(
                context_parallel(attention.inner_attention),
                FLAContextParallelInnerGDN.Config,
                preserve_conv_stride=True,
            )
            layer.attention = derive(attention, _MasterOrderGDN.Config, inner_attention=inner)
    for _, linear, _, _ in config.model.traverse(Linear.Config):
        linear.inplace_wgrad_accum = False
