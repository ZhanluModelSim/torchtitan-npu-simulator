# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatcher-visible Meta contracts for Qwen3.5's NPU GDN kernel and text inputs.

The production Qwen3.5 path binds the Triton-Ascend GatedDeltaRule kernel
(``torchtitan_npu.ops.triton.gdn``); Triton cannot execute on Meta, so the
simulator exposes the same fusion boundary as a virtual custom op with a fake
Meta kernel and an explicit backward op, mirroring Kimi's ``kimi_sim`` fusions.
Real-device calls fail explicitly.

The upstream model owns its input path through ``get_attention_masks`` and the
transitional runner contract expects ``build_attention_masks``; the adapter
below supplies text-only synthetic inputs (no vision tower in the graph).
"""

import torch

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)

_TEXT_SPECIAL_TOKENS = {"image_id": -1, "video_id": -1}

# Axis names mirror the production kernel boundaries: the Triton-Ascend GDN
# contracts (batch, sequence, head, dim) per torchtitan_npu.ops.triton.gdn,
# while flex attention keeps the upstream (batch, head, seq, dim) layout. The
# optional state reset mask is absent on the fixed-length text capture path.
_GDN_INPUTS = (
    ("xq_BLNK", TensorMeaning("query", ("batch", "sequence", "head", "key_head_dim"))),
    ("xk_BLNK", TensorMeaning("key", ("batch", "sequence", "head", "key_head_dim"))),
    ("xv_BLNV", TensorMeaning("value", ("batch", "sequence", "head", "value_head_dim"))),
    ("g_BLN", TensorMeaning("decay_gate", ("batch", "sequence", "head"))),
    ("beta_BLN", TensorMeaning("update_gate", ("batch", "sequence", "head"))),
    ("reset", TensorMeaning("state_reset_mask", ("batch", "sequence"), optional=True)),
)
_GDN_OUTPUT = TensorMeaning("attention_output", ("batch", "sequence", "head", "value_head_dim"))
_FLEX_INPUTS = (
    ("q_BNLH", TensorMeaning("query", ("batch", "head", "q_len", "head_dim"))),
    ("k_BNLH", TensorMeaning("key", ("batch", "head", "kv_len", "head_dim"))),
    ("v_BNLH", TensorMeaning("value", ("batch", "head", "kv_len", "head_dim"))),
)
_FLEX_OUTPUT = TensorMeaning("attention_output", ("batch", "head", "q_len", "head_dim"))


def _register_fusion_semantics(
    forward: torch.library.CustomOpDef,
    backward: torch.library.CustomOpDef,
    inputs: tuple[tuple[str, TensorMeaning], ...],
    output: TensorMeaning,
    backward_inputs: tuple[tuple[str, TensorMeaning], ...],
    dy_name: str,
    differentiable_roles: tuple[str, ...],
) -> None:
    """Declare roles for one forward/backward op pair.

    ``differentiable_roles`` lists the forward argument names that receive a
    gradient at this boundary; non-differentiable optional inputs (the GDN
    state reset mask) keep their declared input meaning without a gradient
    output.
    """
    register_tensor_semantics(str(forward._opoverload), OperatorTensorSemantics(inputs, (output,)))
    output_gradient = TensorMeaning("output_gradient", output.axes, output.unknown_leading_axes)
    meaning_by_name = dict(inputs)
    backward_output_meanings = tuple(
        TensorMeaning(
            f"{meaning_by_name[name].role}_gradient",
            meaning_by_name[name].axes,
            meaning_by_name[name].unknown_leading_axes,
        )
        for name in differentiable_roles
    )
    register_tensor_semantics(
        str(backward._opoverload),
        OperatorTensorSemantics((*backward_inputs, (dy_name, output_gradient)), backward_output_meanings),
    )


@torch.library.custom_op("qwen3_5_sim::gated_delta_rule", mutates_args=())
def gated_delta_rule(
    xq_BLNK: torch.Tensor,
    xk_BLNK: torch.Tensor,
    xv_BLNV: torch.Tensor,
    g_BLN: torch.Tensor,
    beta_BLN: torch.Tensor,
    reset: torch.Tensor | None,
) -> torch.Tensor:
    """Virtual boundary of the Triton-Ascend chunked GatedDeltaRule kernel."""
    raise RuntimeError("Qwen3.5 virtual GDN is available only for Meta simulation")


@gated_delta_rule.register_fake
def _gdn_meta(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, reset):
    return torch.empty_like(xv_BLNV)


@torch.library.custom_op("qwen3_5_sim::gated_delta_rule_grad", mutates_args=())
def gated_delta_rule_grad(
    xq_BLNK: torch.Tensor,
    xk_BLNK: torch.Tensor,
    xv_BLNV: torch.Tensor,
    g_BLN: torch.Tensor,
    beta_BLN: torch.Tensor,
    reset: torch.Tensor | None,
    dy_BLNV: torch.Tensor,
) -> list[torch.Tensor]:
    """Virtual backward of the Triton-Ascend chunked GatedDeltaRule kernel."""
    raise RuntimeError("Qwen3.5 virtual GDN backward is available only for Meta simulation")


@gated_delta_rule_grad.register_fake
def _gdn_grad_meta(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, reset, dy_BLNV):
    return [torch.empty_like(t) for t in (xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN)]


def _gdn_setup(ctx, inputs, output):
    # reset may be None (fixed-length capture); save_for_backward keeps it.
    ctx.save_for_backward(*inputs)
    ctx.has_reset = inputs[5] is not None


def _gdn_backward(ctx, dy):
    xq, xk, xv, g, beta, reset = ctx.saved_tensors
    grads = gated_delta_rule_grad(xq, xk, xv, g, beta, reset, dy)
    return (*grads, None)


gated_delta_rule.register_autograd(_gdn_backward, setup_context=_gdn_setup)


@torch.library.custom_op("qwen3_5_sim::flex_attention", mutates_args=())
def flex_attention(
    q_BNLH: torch.Tensor,
    k_BNLH: torch.Tensor,
    v_BNLH: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """Virtual boundary of one compiled FlexAttention computation.

    The mask/block structure and kernel options do not change the operator
    stream, so the fusion is recorded with the query/key/value tensors and
    the two scalar hyperparameters only.
    """
    raise RuntimeError("Qwen3.5 virtual flex attention is available only for Meta simulation")


@flex_attention.register_fake
def _flex_meta(q_BNLH, k_BNLH, v_BNLH, scale, enable_gqa):
    return torch.empty_like(q_BNLH)


@torch.library.custom_op("qwen3_5_sim::flex_attention_grad", mutates_args=())
def flex_attention_grad(
    q_BNLH: torch.Tensor,
    k_BNLH: torch.Tensor,
    v_BNLH: torch.Tensor,
    dy_BNLH: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> list[torch.Tensor]:
    """Virtual backward of one compiled FlexAttention computation."""
    raise RuntimeError("Qwen3.5 virtual flex attention backward is available only for Meta simulation")


@flex_attention_grad.register_fake
def _flex_grad_meta(q_BNLH, k_BNLH, v_BNLH, dy_BNLH, scale, enable_gqa):
    return [torch.empty_like(t) for t in (q_BNLH, k_BNLH, v_BNLH)]


def _flex_setup(ctx, inputs, output):
    # The bridge passes scale/enable_gqa positionally, so they land in inputs.
    ctx.save_for_backward(*inputs[:3])
    ctx.scale = inputs[3]
    ctx.enable_gqa = inputs[4]


def _flex_backward(ctx, dy):
    q, k, v = ctx.saved_tensors
    dq, dk, dv = flex_attention_grad(q, k, v, dy, ctx.scale, ctx.enable_gqa)
    return dq, dk, dv, None, None


flex_attention.register_autograd(_flex_backward, setup_context=_flex_setup)


_register_fusion_semantics(
    gated_delta_rule,
    gated_delta_rule_grad,
    _GDN_INPUTS,
    _GDN_OUTPUT,
    _GDN_INPUTS,
    "dy_BLNV",
    ("xq_BLNK", "xk_BLNK", "xv_BLNV", "g_BLN", "beta_BLN"),
)
_register_fusion_semantics(
    flex_attention,
    flex_attention_grad,
    _FLEX_INPUTS,
    _FLEX_OUTPUT,
    _FLEX_INPUTS,
    "dy_BNLH",
    ("q_BNLH", "k_BNLH", "v_BNLH"),
)


def _flex_bridge(q_BNLH, k_BNLH, v_BNLH, **kwargs):
    from torchtitan_npu.extensions.torchtitan.eager_flex import _make_aux_output

    # out_transform is None on the Qwen3.5 capture path, so the aux payload
    # is never consumed; return the empty aux to keep the (out, aux) contract.
    out = flex_attention(q_BNLH, k_BNLH, v_BNLH, kwargs["scale"], kwargs["enable_gqa"])
    return out, _make_aux_output()


def _gdn_bridge(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, *, reset=None):
    return gated_delta_rule(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, reset)


def _build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
    """Text-only synthetic input path for the runner's transitional seam.

    Upstream Qwen35Model owns ``get_attention_masks`` (per-consumer dict for
    the hybrid stack); the fixed-length capture has a single document, so no
    CP sharding or load balancer is involved.
    """
    if cp_mesh is not None:
        # CP requires the varlen/asc_cp recipes and rank-local QwenCPMetadata,
        # which are outside the fixed-length text capture scope.
        if load_balancer_type is not None:
            raise NotImplementedError("Qwen3.5 simulator capture does not use a CP load balancer")
        raise NotImplementedError("Qwen3.5 simulator capture does not support context parallel")
    extra = dict(extra_inputs)
    extra.setdefault("special_tokens", dict(_TEXT_SPECIAL_TOKENS))
    extra["attention_masks"] = self.get_attention_masks(extra["positions"])
    return tokens, labels, extra


def install_qwen3_5_meta_support(scope):
    """Replace the Triton GDN and compiled-flex dispatches, bind the input adapter."""
    from torchtitan.models.common.attention import FlexAttention
    from torchtitan.models.qwen3_5.model import Qwen35Model

    from torchtitan_npu.override.qwen3_5 import gated_delta as gdn_owner

    scope.replace(gdn_owner, "run_gdn", _gdn_bridge)
    scope.replace(FlexAttention, "compiled_flex_attn", _flex_bridge)
    scope.replace(Qwen35Model, "build_attention_masks", _build_attention_masks)


__all__ = [
    "flex_attention",
    "flex_attention_grad",
    "gated_delta_rule",
    "gated_delta_rule_grad",
    "install_qwen3_5_meta_support",
]
