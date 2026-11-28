# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatcher-visible Meta contracts for the multi_head_moe SLA2 branch cores.

The model's SLA2 attention has two branch seams
(``run_sla2_sparse_attention`` / ``run_sla2_linear_attention`` in
``torchtitan_npu.models.multi_head_moe.model``). A production NPU build binds
fused kernels there: the sparse branch is a block-sparse attention kernel
(the CANN ``aclnnBlockSparseAttention`` family computes only selected
blocks), the linear branch is a chunked linear-attention kernel (约定生产名,
no public CANN anchor). The eager fallbacks materialize the full score
matrix and per-block states, which misrepresents the architecture's
execution, so the simulator exposes the two fused boundaries as virtual
custom ops with fake Meta kernels and explicit backward ops — the same
pattern as the Kimi and Qwen3.5 fused-boundary fusions.
Real-device calls fail explicitly. Installation is opt-in: the simulator
replaces the seams only when ``SimulationConfig.enable_sla2_fusion`` is set
(default off, eager reference operators).

Everything around the fusion stays native dispatch: block pooling, the block
router projections, ``M_not = 1 - M_c`` and the alpha branch mix. The mask
differentiability matches the eager reference per branch: the sparse branch
consumes its mask through the boolean gate ``masked_fill(mask < 1e-6)`` so its
backward routes None to the router, while the linear branch consumes the
complement mask as a real differentiable einsum operand and its backward
returns the mask gradient. Two training-path elementwise chains also collapse
into virtual fused boundaries under the same gate: the block router's gumbel
soft top-k mask (``_soft_topk_mask``) and the silu-gated activation shared by
the common FeedForward (shared experts) and GroupedExperts (routed experts) —
the gate/up projections and grouped GEMMs stay native dispatch.
"""

import torch

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)

_QKV_AXES = ("batch", "head", "sequence", "head_dim")
_MASK_AXES = ("batch", "head", "q_blocks", "kv_blocks")

_SPARSE_INPUTS = (
    ("q_BHTD", TensorMeaning("query", _QKV_AXES)),
    ("k_BHTD", TensorMeaning("key", _QKV_AXES)),
    ("v_BHTD", TensorMeaning("value", _QKV_AXES)),
    ("block_mask_BHqq", TensorMeaning("block_mask", _MASK_AXES)),
)
_SPARSE_OUTPUT = TensorMeaning("attention_output", _QKV_AXES)
# The linear branch consumes the complement mask M_not = 1 - M_c; it is
# materialized by native dispatch between the two fusions.
_LINEAR_INPUTS = (
    ("q_BHTD", TensorMeaning("query", _QKV_AXES)),
    ("k_BHTD", TensorMeaning("key", _QKV_AXES)),
    ("v_BHTD", TensorMeaning("value", _QKV_AXES)),
    ("complement_mask_BHqq", TensorMeaning("complement_block_mask", _MASK_AXES)),
)
_LINEAR_OUTPUT = TensorMeaning("attention_output", _QKV_AXES)


def _register_fusion_semantics(
    forward: torch.library.CustomOpDef,
    backward: torch.library.CustomOpDef,
    inputs: tuple[tuple[str, TensorMeaning], ...],
    output: TensorMeaning,
    dy_name: str,
    differentiable_roles: tuple[str, ...],
) -> None:
    """Declare roles for one forward/backward op pair.

    ``differentiable_roles`` lists the forward argument names that receive a
    gradient at this boundary; scalar hyperparameters keep their declared
    input meaning without a gradient output.
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
    backward_inputs = (*inputs, (dy_name, output_gradient))
    register_tensor_semantics(
        str(backward._opoverload),
        OperatorTensorSemantics(backward_inputs, backward_output_meanings),
    )


@torch.library.custom_op("multi_head_moe_sim::sla2_sparse_attention", mutates_args=())
def sla2_sparse_attention(
    q_BHTD: torch.Tensor,
    k_BHTD: torch.Tensor,
    v_BHTD: torch.Tensor,
    block_mask_BHqq: torch.Tensor,
    selected_blocks: int,
) -> torch.Tensor:
    """Virtual boundary of the fused SLA2 block-sparse attention kernel.

    ``selected_blocks`` (per query row) records how many key blocks the
    production kernel computes; the cost model reads it from the captured
    node attributes.
    """
    raise RuntimeError("multi_head_moe virtual SLA2 sparse attention is available only for Meta simulation")


@sla2_sparse_attention.register_fake
def _sla2_sparse_meta(q_BHTD, k_BHTD, v_BHTD, block_mask_BHqq, selected_blocks):
    return torch.empty_like(q_BHTD)


@torch.library.custom_op("multi_head_moe_sim::sla2_sparse_attention_grad", mutates_args=())
def sla2_sparse_attention_grad(
    q_BHTD: torch.Tensor,
    k_BHTD: torch.Tensor,
    v_BHTD: torch.Tensor,
    block_mask_BHqq: torch.Tensor,
    selected_blocks: int,
    dy_BHTD: torch.Tensor,
) -> list[torch.Tensor]:
    """Virtual backward of the fused SLA2 block-sparse attention kernel."""
    raise RuntimeError("multi_head_moe virtual SLA2 sparse attention backward is available only for Meta simulation")


@sla2_sparse_attention_grad.register_fake
def _sla2_sparse_grad_meta(q_BHTD, k_BHTD, v_BHTD, block_mask_BHqq, selected_blocks, dy_BHTD):
    return [
        torch.empty_like(q_BHTD),
        torch.empty_like(k_BHTD),
        torch.empty_like(v_BHTD),
        torch.empty_like(block_mask_BHqq),
    ]


def _sparse_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4])
    ctx.selected_blocks = inputs[4]


def _sparse_backward(ctx, dy):
    q, k, v, mask = ctx.saved_tensors
    grads = sla2_sparse_attention_grad(q, k, v, mask, ctx.selected_blocks, dy)
    # Eager consumes the block mask through the boolean gate
    # (``masked_fill(mask < 1e-6)``), so no gradient flows back to the
    # router; the virtual op keeps the fourth output slot for the cost
    # adaptor contract but autograd routes None for the mask, matching the
    # eager reference. None also marks the non-differentiable
    # selected_blocks scalar.
    _dq, _dk, _dv, _dmask = grads
    return (_dq, _dk, _dv, None, None)


sla2_sparse_attention.register_autograd(_sparse_backward, setup_context=_sparse_setup)


@torch.library.custom_op("multi_head_moe_sim::sla2_linear_attention", mutates_args=())
def sla2_linear_attention(
    q_BHTD: torch.Tensor,
    k_BHTD: torch.Tensor,
    v_BHTD: torch.Tensor,
    complement_mask_BHqq: torch.Tensor,
    selected_blocks: int,
) -> torch.Tensor:
    """Virtual boundary of the fused SLA2 chunked linear-attention kernel.

    ``selected_blocks`` records the per-row complement key-block count the
    production kernel accumulates over; the cost model reads it from the
    captured node attributes.
    """
    raise RuntimeError("multi_head_moe virtual SLA2 linear attention is available only for Meta simulation")


@sla2_linear_attention.register_fake
def _sla2_linear_meta(q_BHTD, k_BHTD, v_BHTD, complement_mask_BHqq, selected_blocks):
    return torch.empty_like(q_BHTD)


@torch.library.custom_op("multi_head_moe_sim::sla2_linear_attention_grad", mutates_args=())
def sla2_linear_attention_grad(
    q_BHTD: torch.Tensor,
    k_BHTD: torch.Tensor,
    v_BHTD: torch.Tensor,
    complement_mask_BHqq: torch.Tensor,
    selected_blocks: int,
    dy_BHTD: torch.Tensor,
) -> list[torch.Tensor]:
    """Virtual backward of the fused SLA2 chunked linear-attention kernel."""
    raise RuntimeError("multi_head_moe virtual SLA2 linear attention backward is available only for Meta simulation")


@sla2_linear_attention_grad.register_fake
def _sla2_linear_grad_meta(q_BHTD, k_BHTD, v_BHTD, complement_mask_BHqq, selected_blocks, dy_BHTD):
    return [
        torch.empty_like(q_BHTD),
        torch.empty_like(k_BHTD),
        torch.empty_like(v_BHTD),
        torch.empty_like(complement_mask_BHqq),
    ]


def _linear_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4])
    ctx.selected_blocks = inputs[4]


def _linear_backward(ctx, dy):
    q, k, v, mask = ctx.saved_tensors
    grads = sla2_linear_attention_grad(q, k, v, mask, ctx.selected_blocks, dy)
    # None marks the non-differentiable selected_blocks scalar.
    return (*grads, None)


sla2_linear_attention.register_autograd(_linear_backward, setup_context=_linear_setup)


_register_fusion_semantics(
    sla2_sparse_attention,
    sla2_sparse_attention_grad,
    _SPARSE_INPUTS,
    _SPARSE_OUTPUT,
    "dy_BHTD",
    ("q_BHTD", "k_BHTD", "v_BHTD", "block_mask_BHqq"),
)
_register_fusion_semantics(
    sla2_linear_attention,
    sla2_linear_attention_grad,
    _LINEAR_INPUTS,
    _LINEAR_OUTPUT,
    "dy_BHTD",
    ("q_BHTD", "k_BHTD", "v_BHTD", "complement_mask_BHqq"),
)


def _sparse_bridge(q_BHTD, k_BHTD, v_BHTD, block_mask_BHqq, selected_blocks):
    return sla2_sparse_attention(q_BHTD, k_BHTD, v_BHTD, block_mask_BHqq, selected_blocks)


def _linear_bridge(q_BHTD, k_BHTD, v_BHTD, complement_mask_BHqq, selected_blocks):
    return sla2_linear_attention(q_BHTD, k_BHTD, v_BHTD, complement_mask_BHqq, selected_blocks)


_SWIGLU_AXES = ("feature",)
_SWIGLU_INPUTS = (
    ("x", TensorMeaning("x", _SWIGLU_AXES, unknown_leading_axes=True)),
    ("gate", TensorMeaning("gate", _SWIGLU_AXES, unknown_leading_axes=True)),
)
_SWIGLU_OUTPUT = TensorMeaning("activated_output", _SWIGLU_AXES, unknown_leading_axes=True)


@torch.library.custom_op("multi_head_moe_sim::swiglu", mutates_args=())
def moe_sim_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Virtual boundary of the fused silu-gated activation kernel.

    ``silu(gate) * x`` in one vector kernel: the shared-expert FeedForward and
    the routed GroupedExperts both compute this activation between their gate
    and up projections. The projections and grouped GEMMs stay native dispatch.
    """
    raise RuntimeError("multi_head_moe virtual swiglu is available only for Meta simulation")


@moe_sim_swiglu.register_fake
def _moe_sim_swiglu_meta(x, gate):
    return torch.empty_like(x)


@torch.library.custom_op("multi_head_moe_sim::swiglu_backward", mutates_args=())
def moe_sim_swiglu_backward(x: torch.Tensor, gate: torch.Tensor, dy: torch.Tensor) -> list[torch.Tensor]:
    """Virtual backward of the fused silu-gated activation kernel."""
    raise RuntimeError("multi_head_moe virtual swiglu backward is available only for Meta simulation")


@moe_sim_swiglu_backward.register_fake
def _moe_sim_swiglu_backward_meta(x, gate, dy):
    return [torch.empty_like(x), torch.empty_like(gate)]


def _moe_sim_swiglu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _moe_sim_swiglu_backward(ctx, dy):
    x, gate = ctx.saved_tensors
    return (*moe_sim_swiglu_backward(x, gate, dy),)


moe_sim_swiglu.register_autograd(_moe_sim_swiglu_backward, setup_context=_moe_sim_swiglu_setup)

_register_fusion_semantics(moe_sim_swiglu, moe_sim_swiglu_backward, _SWIGLU_INPUTS, _SWIGLU_OUTPUT, "dy", ("x", "gate"))


@torch.library.custom_op("multi_head_moe_sim::router_soft_topk", mutates_args=())
def router_soft_topk(score_BHqq: torch.Tensor, temp: float, topk_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Virtual boundary of the fused gumbel soft top-k block-mask kernel.

    Training-path fusion of the reference ``_soft_topk_mask`` chain
    (``sigmoid((softmax(score/temp + gumbel) - topk_threshold)/temp)``): the
    gumbel noise draw, softmax reduction, top-k threshold selection and
    sigmoid smoothing run inside one kernel. It returns the block mask plus
    the softmax probabilities the backward consumes (the threshold subgradient
    is zero, matching a straight-through read of the k-th value). Real-device
    calls fail explicitly.
    """
    raise RuntimeError("multi_head_moe virtual router soft top-k is available only for Meta simulation")


@router_soft_topk.register_fake
def _router_soft_topk_meta(score_BHqq, temp, topk_blocks):
    return torch.empty_like(score_BHqq), torch.empty_like(score_BHqq)


@torch.library.custom_op("multi_head_moe_sim::router_soft_topk_backward", mutates_args=())
def router_soft_topk_backward(
    mask_BHqq: torch.Tensor,
    s_BHqq: torch.Tensor,
    dy_BHqq: torch.Tensor,
    temp: float,
) -> list[torch.Tensor]:
    """Virtual backward of the fused gumbel soft top-k block-mask kernel."""
    raise RuntimeError("multi_head_moe virtual router soft top-k backward is available only for Meta simulation")


@router_soft_topk_backward.register_fake
def _router_soft_topk_backward_meta(mask_BHqq, s_BHqq, dy_BHqq, temp):
    return [torch.empty_like(mask_BHqq)]


def _router_setup(ctx, inputs, output):
    ctx.save_for_backward(output[0], output[1])
    ctx.temp = inputs[1]


def _router_backward(ctx, dy_mask, _dy_s):
    mask, s = ctx.saved_tensors
    return (*router_soft_topk_backward(mask, s, dy_mask, ctx.temp), None, None)


router_soft_topk.register_autograd(_router_backward, setup_context=_router_setup)

register_tensor_semantics(
    str(router_soft_topk._opoverload),
    OperatorTensorSemantics(
        (("score_BHqq", TensorMeaning("score", _MASK_AXES)),),
        (
            TensorMeaning("block_mask", _MASK_AXES),
            TensorMeaning("softmax_output", _MASK_AXES),
        ),
    ),
)
register_tensor_semantics(
    str(router_soft_topk_backward._opoverload),
    OperatorTensorSemantics(
        (
            ("mask_BHqq", TensorMeaning("block_mask", _MASK_AXES)),
            ("s_BHqq", TensorMeaning("softmax_output", _MASK_AXES)),
            ("dy_BHqq", TensorMeaning("output_gradient", _MASK_AXES)),
        ),
        (TensorMeaning("score_gradient", _MASK_AXES),),
    ),
)


def _rotary_bridge(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Fused interleave RoPE via ``torch_npu.npu_rotary_mul``.

    The model's local ``apply_rotary_emb`` is a plain function (not a
    Configurable), so the shared ``override.common.rope.asc_complex`` —
    installed for multi_head_moe — cannot reach it and the capture ran the
    decomposed complex-mul math (``view_as_complex``/``view_as_real`` are
    unmodelled and silently zero-cost). Production binds the same AscendC
    interleave kernel the DSV4 families use; ``install_rotary_bridge`` (runtime
    setup) already wraps ``torch_npu.npu_rotary_mul`` with the Meta autograd
    bridge, so the fake device dispatches the fused op directly. Numerics match
    the eager reference: fp32-internal rotation (``x.float()`` upcast), cast
    back to the input dtype; ``freqs_cis`` is the polar (cos + i*sin) cache.
    """
    import torch_npu

    # The AscendC interleave kernel takes full-width cos/sin factors: each
    # pair's value (cos_p + i*sin_p from the polar cache) occupies both
    # elements 2p and 2p+1, so duplicate the half-width cache and broadcast
    # over (batch, seq, head) against the activation — OPM's rotary semantics
    # require factors that broadcast to the full activation shape.
    cos = torch.repeat_interleave(freqs_cis.real, 2, dim=-1)
    sin = torch.repeat_interleave(freqs_cis.imag, 2, dim=-1)
    cos = cos.reshape(1, cos.shape[0], 1, -1).to(x.dtype)
    sin = sin.reshape(1, sin.shape[0], 1, -1).to(x.dtype)
    out = torch_npu.npu_rotary_mul(x.float(), cos, sin, rotary_mode="interleave")
    return out.type_as(x)


def _soft_topk_bridge(self, score_BHqq, topk_blocks):
    """Fused training-path gumbel soft top-k mask; the eval hard top-k stays native."""
    if self.training:
        mask_BHqq, _s_BHqq = router_soft_topk(score_BHqq, self.temp, topk_blocks)
        return mask_BHqq
    _, top_idx = torch.topk(score_BHqq, k=topk_blocks, dim=-1)
    mask = torch.zeros_like(score_BHqq)
    mask.scatter_(-1, top_idx, 1.0)
    return mask


def _feed_forward_bridge(self, x):
    return self.w2(moe_sim_swiglu(self.w3(x), self.w1(x)))


def _grouped_experts_bridge(self, x_RD, num_tokens_per_expert_E, *, routed_scores_R=None):
    import spmd_types as spmd
    from torch.distributed.tensor import DTensor
    from torchtitan.distributed.spmd_types import spmd_mesh_size
    from torchtitan.distributed.utils import get_spmd_backend

    if isinstance(self.w1_EFD, DTensor):
        w1 = self.w1_EFD.to_local()
        assert isinstance(self.w2_EDF, DTensor)
        w2 = self.w2_EDF.to_local()
        assert isinstance(self.w3_EFD, DTensor)
        w3 = self.w3_EFD.to_local()
    else:
        w1 = self.w1_EFD
        w2 = self.w2_EDF
        w3 = self.w3_EFD

    offsets = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
        for axis in ("dp", "cp"):
            spmd.mutate_type(offsets, axis, src=spmd.P, dst=spmd.V)

    gate = self._grouped_mm(A=x_RD.bfloat16(), B_t=w1.bfloat16().transpose(-2, -1), offs=offsets)
    up = self._grouped_mm(A=x_RD.bfloat16(), B_t=w3.bfloat16().transpose(-2, -1), offs=offsets)
    hidden = moe_sim_swiglu(up, gate)
    return self._grouped_mm(A=hidden, B_t=w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x_RD)


def install_multi_head_moe_meta_support(scope):
    """Replace the eager SLA2 branch fallbacks and the eager activations with the virtual fused ops."""
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.common.moe import GroupedExperts

    from torchtitan_npu.models.multi_head_moe import model as owner
    from torchtitan_npu.models.multi_head_moe.model import InnerBlockSLA2

    scope.replace(owner, "run_sla2_sparse_attention", _sparse_bridge)
    scope.replace(owner, "run_sla2_linear_attention", _linear_bridge)
    scope.replace(owner, "apply_rotary_emb", _rotary_bridge)
    scope.replace(InnerBlockSLA2, "_soft_topk_mask", _soft_topk_bridge)
    scope.replace(FeedForward, "forward", _feed_forward_bridge)
    scope.replace(GroupedExperts, "forward", _grouped_experts_bridge)


__all__ = [
    "install_multi_head_moe_meta_support",
    "moe_sim_swiglu",
    "moe_sim_swiglu_backward",
    "router_soft_topk",
    "router_soft_topk_backward",
    "sla2_linear_attention",
    "sla2_linear_attention_grad",
    "sla2_sparse_attention",
    "sla2_sparse_attention_grad",
]
