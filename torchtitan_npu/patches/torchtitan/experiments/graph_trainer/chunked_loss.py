# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4734

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coalesce GraphTrainer chunk-loss gradients for one-dimensional SimpleFSDP.

Upstream review requests a model/loss-level fix instead of a dedicated graph
pass. This temporary implementation leaves unsupported layouts on the original
path; removal requires equivalent upstream SimpleFSDP support.
Remove this module after the TorchTitan dependency includes the PR.
"""

from functools import wraps
from inspect import getclosurevars, isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtitan.experiments.graph_trainer.chunked_loss
from torch.distributed.tensor import DTensor, Partial, Shard
from torchtitan.experiments.graph_trainer.chunked_loss import ChunkedLossWrapperWithParamGrads
from torchtitan.experiments.graph_trainer.simple_fsdp import ReplicateComputation
from torchtitan.tools.logging import logger

_original_call = ChunkedLossWrapperWithParamGrads.__call__
_original_gradient_backprop = ChunkedLossWrapperWithParamGrads._gradient_backprop


class _LocalGradientLMHead(nn.Module):
    sharded_weight: DTensor

    def __init__(self, weight: torch.Tensor, sharded_weight: DTensor):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=sharded_weight.requires_grad)
        # Keep the original parameter out of this temporary module's parameters.
        object.__setattr__(self, "sharded_weight", sharded_weight)
        self.accumulated_grad: torch.Tensor | None = None
        self.grad_hook = self.weight.register_post_accumulate_grad_hook(self._accumulate_grad)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)

    def _accumulate_grad(self, weight: torch.Tensor) -> None:
        assert weight.grad is not None
        grad = weight.grad.to(torch.float32)
        if self.accumulated_grad is None:
            self.accumulated_grad = grad
        else:
            self.accumulated_grad.add_(grad)
        # Never accumulate successive chunks in the BF16 compute parameter.
        weight.grad = None


class _ChunkGradsBackprop(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, hidden, weight, hidden_grad, weight_grad, loss):
        ctx.save_for_backward(hidden_grad, weight_grad)
        ctx.mesh = weight.device_mesh
        ctx.placements = weight.placements
        ctx.weight_shape = weight.shape
        ctx.weight_stride = weight.stride()
        return loss.detach().clone()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        hidden_grad, weight_grad = ctx.saved_tensors
        # Keep the collective in backward so GraphTrainer's bucketing and
        # scheduling retain the same gradient-communication metadata.
        partial_grad = DTensor.from_local(
            weight_grad,
            ctx.mesh,
            (Partial("sum"),),
            shape=ctx.weight_shape,
            stride=ctx.weight_stride,
            run_check=False,
        )
        sharded_grad = partial_grad.redistribute(placements=ctx.placements)
        return hidden_grad * grad_output, sharded_grad * grad_output, None, None, None


def _can_coalesce(head, pred) -> bool:
    if (
        not isinstance(head, nn.Linear)
        or type(head).forward is not nn.Linear.forward
        or head.bias is not None
        or isinstance(pred, DTensor)
    ):
        return False
    weight = head._parameters.get("weight")
    if (
        not isinstance(weight, DTensor)
        or weight.device_mesh.ndim != 1
        or weight.placements != (Shard(0),)
        or weight.dtype != torch.float32
        or not weight.requires_grad
    ):
        return False

    # SimpleFSDP v0.3.0 keeps its precision policy in the weight getter closure.
    weight_property = getattr(type(head), "weight", None)
    if not isinstance(weight_property, property) or not isfunction(weight_property.fget):
        return False
    parametrization = getclosurevars(weight_property.fget).nonlocals.get("parametrization")
    if not isinstance(parametrization, ReplicateComputation) or parametrization.full_dtensor:
        return False
    reduce_dtype = parametrization.reduce_dtype or parametrization.param_dtype or weight.dtype
    return reduce_dtype == torch.float32


@wraps(_original_call)
def _coalesced_call(self, pred, labels, global_valid_tokens=None, **loss_inputs):
    head = self.lm_head
    if type(self) is not ChunkedLossWrapperWithParamGrads or not pred.requires_grad or head is None:
        return _original_call(self, pred, labels, global_valid_tokens, **loss_inputs)
    supported = _can_coalesce(head, pred)
    if not getattr(self, "_coalesced_rs_logged", False):
        if supported:
            logger.info("[PATCH] GraphTrainer chunked loss: coalescing SimpleFSDP gradients before reduce-scatter")
        else:
            logger.warning("[PATCH] Chunk-loss RS coalescing is unsupported for this head/layout; using upstream loss")
        self._coalesced_rs_logged = True  # pyrefly: ignore[missing-attribute]
    if not supported:
        return _original_call(self, pred, labels, global_valid_tokens, **loss_inputs)

    # Preserve the original all-gather and compute dtype, but defer gradient sync.
    local_head = _LocalGradientLMHead(head.weight, head._parameters["weight"])
    self.lm_head = local_head
    try:
        return _original_call(self, pred, labels, global_valid_tokens, **loss_inputs)
    finally:
        local_head.grad_hook.remove()
        self.lm_head = head


def _gradient_backprop(hidden_states, accumulated_grad, total_loss, lm_head, fsdp_enabled):
    if not isinstance(lm_head, _LocalGradientLMHead):
        return _original_gradient_backprop(hidden_states, accumulated_grad, total_loss, lm_head, fsdp_enabled)
    assert not fsdp_enabled
    assert lm_head.accumulated_grad is not None
    # Partial SUM is a local contribution; do not divide by num_chunks again.
    return _ChunkGradsBackprop.apply(
        hidden_states,
        lm_head.sharded_weight,
        accumulated_grad,
        lm_head.accumulated_grad,
        total_loss,
    )


def apply() -> None:
    # Patch methods in place so previously imported classes/configs also see it.
    torchtitan.experiments.graph_trainer.chunked_loss.ChunkedLossWrapperWithParamGrads.__call__ = _coalesced_call
    torchtitan.experiments.graph_trainer.chunked_loss.ChunkedLossWrapperWithParamGrads._gradient_backprop = (
        staticmethod(_gradient_backprop)
    )


apply()
