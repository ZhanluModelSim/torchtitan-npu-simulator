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

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)


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


# These layouts are part of the fusion boundary in KimiDeltaAttention and
# KimiGatedMLA, not deductions from tensor sizes. MLA values are already padded.
_KDA_INPUTS = (
    ("q", TensorMeaning("query", ("batch", "sequence", "head", "key_head_dim"))),
    ("k", TensorMeaning("key", ("batch", "sequence", "head", "key_head_dim"))),
    ("v", TensorMeaning("value", ("batch", "sequence", "head", "value_head_dim"))),
    ("g", TensorMeaning("decay_gate", ("batch", "sequence", "head", "key_head_dim"))),
    ("beta", TensorMeaning("update_gate", ("batch", "sequence", "head"))),
    ("A_log", TensorMeaning("log_decay_parameter", ("head",))),
    ("dt_bias", TensorMeaning("time_step_bias", (("head", "key_head_dim"),))),
)
_KDA_OUTPUT = TensorMeaning("attention_output", ("batch", "sequence", "head", "value_head_dim"))
_ATTENTION_INPUTS = (
    ("q", TensorMeaning("query", ("batch", "head", "q_len", "qk_head_dim"))),
    ("k", TensorMeaning("key", ("batch", "head", "kv_len", "qk_head_dim"))),
    ("v", TensorMeaning("value", ("batch", "head", "kv_len", "padded_value_head_dim"))),
)
_ATTENTION_OUTPUT = TensorMeaning("attention_output", ("batch", "head", "q_len", "padded_value_head_dim"))
# SiTU accepts both dense and routed expert rows. Only the final feature axis
# is defined at this boundary; rank does not identify batch or token axes.
_SITU_INPUTS = (
    ("gate", TensorMeaning("gate", ("feature",), unknown_leading_axes=True)),
    ("up", TensorMeaning("up", ("feature",), unknown_leading_axes=True)),
)
_SITU_OUTPUT = TensorMeaning("activated_output", ("feature",), unknown_leading_axes=True)


def _register_fusion_semantics(
    forward: torch.library.CustomOpDef,
    backward: torch.library.CustomOpDef,
    inputs: tuple[tuple[str, TensorMeaning], ...],
    output: TensorMeaning,
) -> None:
    register_tensor_semantics(str(forward._opoverload), OperatorTensorSemantics(inputs, (output,)))
    output_gradient = TensorMeaning("output_gradient", output.axes, output.unknown_leading_axes)
    gradients = tuple(
        TensorMeaning(f"{meaning.role}_gradient", meaning.axes, meaning.unknown_leading_axes) for _, meaning in inputs
    )
    register_tensor_semantics(
        str(backward._opoverload), OperatorTensorSemantics((*inputs, ("dy", output_gradient)), gradients)
    )


_register_fusion_semantics(chunk_kda, chunk_kda_grad, _KDA_INPUTS, _KDA_OUTPUT)
_register_fusion_semantics(fusion_attention, fusion_attention_grad, _ATTENTION_INPUTS, _ATTENTION_OUTPUT)
_register_fusion_semantics(situ_glu, situ_glu_backward, _SITU_INPUTS, _SITU_OUTPUT)


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
