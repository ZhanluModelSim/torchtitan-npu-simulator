# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatcher-visible Meta contracts for Kimi's retained virtual fusions.

These operators describe the legacy simulator boundary, not executable NPU
kernels or their internal workspaces. Real-device calls fail explicitly.
"""

import torch
from torch.distributed.tensor import DTensor


@torch.library.custom_op("kimi_sim::chunk_kda", mutates_args=())
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    raise RuntimeError("Kimi virtual KDA is available only for Meta simulation")


@chunk_kda.register_fake
def _kda_meta(q, k, v, g, beta, A_log, dt_bias, lower_bound):
    return torch.empty_like(v)


@torch.library.custom_op("kimi_sim::chunk_kda_grad", mutates_args=())
def chunk_kda_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    dy: torch.Tensor,
    lower_bound: float | None,
) -> list[torch.Tensor]:
    raise RuntimeError("Kimi virtual KDA backward is available only for Meta simulation")


@chunk_kda_grad.register_fake
def _kda_grad_meta(q, k, v, g, beta, A_log, dt_bias, dy, lower_bound):
    return [torch.empty_like(t) for t in (q, k, v, g, beta, A_log, dt_bias)]


def _kda_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:7])
    ctx.lower_bound = inputs[7]


def _kda_backward(ctx, dy):
    return (*chunk_kda_grad(*ctx.saved_tensors, dy, ctx.lower_bound), None)


chunk_kda.register_autograd(_kda_backward, setup_context=_kda_setup)


@torch.library.custom_op("kimi_sim::fusion_attention", mutates_args=())
def fusion_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    raise RuntimeError("Kimi virtual attention is available only for Meta simulation")


@fusion_attention.register_fake
def _attention_meta(q, k, v, scale):
    return torch.empty_like(q)


@torch.library.custom_op("kimi_sim::fusion_attention_grad", mutates_args=())
def fusion_attention_grad(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dy: torch.Tensor, scale: float
) -> list[torch.Tensor]:
    raise RuntimeError("Kimi virtual attention backward is available only for Meta simulation")


@fusion_attention_grad.register_fake
def _attention_grad_meta(q, k, v, dy, scale):
    return [torch.empty_like(t) for t in (q, k, v)]


def _attention_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3])
    ctx.scale = inputs[3]


def _attention_backward(ctx, dy):
    return (*fusion_attention_grad(*ctx.saved_tensors, dy, ctx.scale), None)


fusion_attention.register_autograd(_attention_backward, setup_context=_attention_setup)


@torch.library.custom_op("kimi_sim::situ_glu", mutates_args=())
def situ_glu(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    raise RuntimeError("Kimi virtual SiTU is available only for Meta simulation")


@situ_glu.register_fake
def _situ_meta(gate, up, beta, linear_beta):
    return torch.empty_like(gate)


@torch.library.custom_op("kimi_sim::situ_glu_backward", mutates_args=())
def situ_glu_backward(
    gate: torch.Tensor, up: torch.Tensor, dy: torch.Tensor, beta: float, linear_beta: float | None
) -> list[torch.Tensor]:
    raise RuntimeError("Kimi virtual SiTU backward is available only for Meta simulation")


@situ_glu_backward.register_fake
def _situ_backward_meta(gate, up, dy, beta, linear_beta):
    return [torch.empty_like(gate), torch.empty_like(up)]


def _situ_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:2])
    ctx.beta, ctx.linear_beta = inputs[2:]


def _situ_backward(ctx, dy):
    return (*situ_glu_backward(*ctx.saved_tensors, dy, ctx.beta, ctx.linear_beta), None, None)


situ_glu.register_autograd(_situ_backward, setup_context=_situ_setup)


def install_kimi_virtual_fusions(scope):
    """Replace only the three legacy fusion call sites, restoring them on exit."""
    from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention, KimiGatedMLA
    from torchtitan_npu.models.kimi_k3.feed_forward import SituGLU

    def kda(module, q, k, v, g, beta):
        a = module.A_log.to_local() if isinstance(module.A_log, DTensor) else module.A_log
        bias = module.dt_bias.to_local() if isinstance(module.dt_bias, DTensor) else module.dt_bias
        return chunk_kda(q, k, v, g, beta, a, bias, module.gate_lower_bound)

    def attention(module, q, k, v):
        return fusion_attention(q, k, v, module.scaling)

    def activation(module, packed):
        gate, up = packed.chunk(2, dim=-1)
        return situ_glu(gate, up, module.beta, module.linear_beta)

    scope.replace(KimiDeltaAttention, "_chunk_kda", kda)
    scope.replace(KimiGatedMLA, "_attention_core", attention)
    scope.replace(SituGLU, "forward", activation)
