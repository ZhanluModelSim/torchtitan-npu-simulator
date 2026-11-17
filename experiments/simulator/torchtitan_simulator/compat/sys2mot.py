# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatcher-visible Meta contract for sys2mot's NPU FlashAttention boundary.

The ViT blocks call ``F.scaled_dot_product_attention`` inline.  On real NPU
devices torch_npu registers an explicit ``PrivateUse1`` kernel for that
operator (``RegisterNPU.cpp``) which dispatches the fused
``aclnnFlashAttentionScore`` kernels, so the production execution is one fused
boundary per attention call.  Meta has only the composite math kernel, which
decomposes into bmm/softmax and hides the fusion.  This module mirrors the
production fusion as a virtual custom op with a fake Meta kernel and an
explicit backward op, following the ``qwen3_5_sim`` flex-attention pattern.
The bridge is installed only for simulator capture sessions; real-device
training never sees it.

RMSNorm and rotary-apply bridging deliberately deviates from production: the
sys2mot reference implementation expresses both as unfused eager chains and the
measured NPU run launches one aclnn kernel per chain member (no aclnnRmsNorm /
fused rotary in the kernel trace).  Capture sessions route them through the
vendor fused operators (``npu.npu_rms_norm`` / ``npu.npu_rotary_mul``) to keep
the captured graph compact; per-op correspondence with the production trace is
intentionally surrendered for these chains.  Non-fp32 inputs and partial or
odd rotation widths keep the reference chain.
"""

import torch

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)

_SDPA_INPUTS = (
    ("query_BNTH", TensorMeaning("query", ("batch", "head", "q_len", "head_dim"))),
    ("key_BNTH", TensorMeaning("key", ("batch", "head", "kv_len", "head_dim"))),
    ("value_BNTH", TensorMeaning("value", ("batch", "head", "kv_len", "head_dim"))),
)
_SDPA_OUTPUT = TensorMeaning("attention_output", ("batch", "head", "q_len", "head_dim"))


def _register_sdpa_semantics(
    forward: torch.library.CustomOpDef,
    backward: torch.library.CustomOpDef,
) -> None:
    """Declare the query/key/value/output roles for one fused FA pair."""
    register_tensor_semantics(str(forward._opoverload), OperatorTensorSemantics(_SDPA_INPUTS, (_SDPA_OUTPUT,)))
    output_gradient = TensorMeaning("output_gradient", _SDPA_OUTPUT.axes, _SDPA_OUTPUT.unknown_leading_axes)
    backward_output_meanings = tuple(
        TensorMeaning(f"{meaning.role}_gradient", meaning.axes, meaning.unknown_leading_axes)
        for _, meaning in _SDPA_INPUTS
    )
    register_tensor_semantics(
        str(backward._opoverload),
        OperatorTensorSemantics((*_SDPA_INPUTS, ("dy_BNTH", output_gradient)), backward_output_meanings),
    )


@torch.library.custom_op("sys2mot_sim::flash_attention", mutates_args=())
def flash_attention(
    query_BNTH: torch.Tensor,
    key_BNTH: torch.Tensor,
    value_BNTH: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """Virtual boundary of the NPU fused FlashAttentionScore kernel."""
    raise RuntimeError("sys2mot virtual FlashAttention is available only for Meta simulation")


@flash_attention.register_fake
def _flash_meta(query_BNTH, key_BNTH, value_BNTH, scale, enable_gqa):
    return torch.empty_like(query_BNTH)


@torch.library.custom_op("sys2mot_sim::flash_attention_grad", mutates_args=())
def flash_attention_grad(
    query_BNTH: torch.Tensor,
    key_BNTH: torch.Tensor,
    value_BNTH: torch.Tensor,
    dy_BNTH: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> list[torch.Tensor]:
    """Virtual backward of the NPU fused FlashAttentionScoreGrad kernel."""
    raise RuntimeError("sys2mot virtual FlashAttention backward is available only for Meta simulation")


@flash_attention_grad.register_fake
def _flash_grad_meta(query_BNTH, key_BNTH, value_BNTH, dy_BNTH, scale, enable_gqa):
    return [torch.empty_like(query_BNTH), torch.empty_like(key_BNTH), torch.empty_like(value_BNTH)]


def _flash_setup(ctx, inputs, output):
    ctx.scale = inputs[3]
    ctx.enable_gqa = inputs[4]
    ctx.save_for_backward(*inputs[:3])


def _flash_backward(ctx, dy):
    query, key, value = ctx.saved_tensors
    grads = flash_attention_grad(query, key, value, dy, ctx.scale, ctx.enable_gqa)
    return (*grads, None, None)


flash_attention.register_autograd(_flash_backward, setup_context=_flash_setup)

_register_sdpa_semantics(flash_attention, flash_attention_grad)


def _sdpa_bridge(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Capture-session stand-in for ``F.scaled_dot_product_attention``.

    sys2mot's ViT calls the operator with plain q/k/v and defaults elsewhere;
    any other shape (mask, dropout, causal) has no fused NPU boundary here and
    fails explicitly instead of being silently remodeled.
    """
    if attn_mask is not None or dropout_p or is_causal:
        raise RuntimeError(
            "sys2mot virtual FlashAttention covers only the maskless non-causal ViT call; "
            f"got attn_mask={attn_mask is not None}, dropout_p={dropout_p}, is_causal={is_causal}"
        )
    if scale is None:
        scale = query.size(-1) ** -0.5
    return flash_attention(query, key, value, scale, enable_gqa)


def _install_sys2mot_norm_rotary(scope) -> dict[str, str]:
    """Route sys2mot RMSNorm and rotary-apply through vendor fused kernels.

    Only float32 full-precision calls are covered; everything else keeps the
    reference eager chain.  Frequency/angle computation stays verbatim so the
    captured cos/sin kernels keep matching the production trace.
    """
    import torch_npu

    from torchtitan_npu.models.sys2mot import model as sys2mot_model
    from torchtitan_simulator.compat.norm import install_meta_norm_autograd
    from torchtitan_simulator.compat.rotary import install_rotary_bridge

    status = {
        "rms_norm_meta_autograd": install_meta_norm_autograd(scope),
        "rotary_meta_autograd": str(install_rotary_bridge(scope).get("rotary_meta_autograd", "native")),
    }
    reference_rms_forward = sys2mot_model.RMSNorm.forward
    reference_rotary = sys2mot_model.rotary

    def rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype is not torch.float32:
            return reference_rms_forward(self, x)
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]

    def rotary_forward(x, pos, dim, theta=10000.0, inverse=False):
        if x.dtype is not torch.float32 or dim % 2 or dim > x.size(-1):
            return reference_rotary(x, pos, dim, theta, inverse)
        freq = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
        angles = pos.float()[..., None] * freq
        angles = angles[None, :, None] if pos.ndim == 1 else angles[:, :, None]
        if inverse:
            angles = -angles
        cosine, sine = angles.cos(), angles.sin()
        tail_input = x if dim == x.size(-1) else x[..., -dim:]
        rotated = torch_npu.npu_rotary_mul(tail_input, cosine, sine, rotary_mode="half")
        if dim == x.size(-1):
            return rotated
        return torch.cat((x[..., :-dim], rotated), -1)

    scope.replace(sys2mot_model.RMSNorm, "forward", rms_norm_forward)
    scope.replace(sys2mot_model, "rotary", rotary_forward)
    status["rms_norm_capture_boundary"] = "npu.npu_rms_norm"
    status["rotary_capture_boundary"] = "npu.npu_rotary_mul[half]"
    return status


def install_sys2mot_meta_support(scope):
    """Install all capture-session Meta bridges for the sys2mot model."""
    import torch.nn.functional as functional

    scope.replace(functional, "scaled_dot_product_attention", _sdpa_bridge)
    return _install_sys2mot_norm_rotary(scope)


__all__ = [
    "flash_attention",
    "flash_attention_grad",
    "install_sys2mot_meta_support",
]
