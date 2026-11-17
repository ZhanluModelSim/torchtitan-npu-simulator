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
pattern as Kimi's ``kimi_sim`` and Qwen3.5's ``qwen3_5_sim`` fusions.
Real-device calls fail explicitly. Installation is opt-in: the simulator
replaces the seams only when ``SimulationConfig.enable_sla2_fusion`` is set
(default off, eager reference operators).

Everything around the fusion stays native dispatch: block pooling, the block
router (its projections are real parameters), the gumbel soft top-k mask,
``M_not = 1 - M_c`` and the alpha branch mix. The block mask is a real
differentiable input of both fusions on the training path, so their
backwards return its gradient alongside the query/key/value gradients.
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
    # None marks the non-differentiable selected_blocks scalar.
    return (*grads, None)


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


def install_multi_head_moe_meta_support(scope):
    """Replace the eager SLA2 branch fallbacks with the virtual fused ops."""
    from torchtitan_npu.models.multi_head_moe import model as owner

    scope.replace(owner, "run_sla2_sparse_attention", _sparse_bridge)
    scope.replace(owner, "run_sla2_linear_attention", _linear_bridge)


__all__ = [
    "install_multi_head_moe_meta_support",
    "sla2_linear_attention",
    "sla2_linear_attention_grad",
    "sla2_sparse_attention",
    "sla2_sparse_attention_grad",
]
