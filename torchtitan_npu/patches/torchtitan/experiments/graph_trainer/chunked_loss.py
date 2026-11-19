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

import logging
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.fx.traceback import annotate
from torchtitan.experiments.graph_trainer.chunked_loss import ChunkedLossWrapperWithParamGrads
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    FSDP_MESH_AXIS_NAMES_META,
    FSDP_PARAM_FQNS_META,
    ReplicateComputation,
)
from torchtitan.models.common.linear import Linear
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor

logger = logging.getLogger(__name__)


class _LocalGradientLMHead(nn.Module):
    sharded_weight: DTensor

    def __init__(self, weight: torch.Tensor, sharded_weight: DTensor, parametrization: ReplicateComputation):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=sharded_weight.requires_grad)
        # Keep the original parameter out of this temporary module's parameters.
        object.__setattr__(self, "sharded_weight", sharded_weight)
        self.parametrization = parametrization
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
    def forward(ctx, hidden, weight, hidden_grad, weight_grad, loss, parametrization):
        ctx.save_for_backward(hidden_grad, weight_grad)
        ctx.mesh = weight.device_mesh
        ctx.placements = weight.placements
        ctx.weight_shape = weight.shape
        ctx.weight_stride = weight.stride()
        ctx.fsdp_meta = {
            FSDP_PARAM_FQNS_META: (parametrization.param_fqn,),
            FSDP_MESH_AXIS_NAMES_META: parametrization.mesh_axis_names,
        }
        return loss.detach().clone()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        hidden_grad, weight_grad = ctx.saved_tensors
        # Keep the collective in backward so GraphTrainer's bucketing and
        # scheduling retain the same gradient-communication metadata.
        with annotate(ctx.fsdp_meta):
            partial_grad = DTensor.from_local(
                weight_grad,
                ctx.mesh,
                (Partial("sum"),),
                shape=ctx.weight_shape,
                stride=ctx.weight_stride,
                run_check=False,
            )
            sharded_grad = partial_grad.redistribute(placements=ctx.placements)
        return hidden_grad * grad_output, sharded_grad * grad_output, None, None, None, None


def _coalescible_parametrization(head, pred) -> ReplicateComputation | None:
    if (
        not isinstance(head, Linear)
        or type(head).forward is not Linear.forward
        or type(head)._linear is not Linear._linear
        or head.num_linears != 1
        or head.inplace_wgrad_accum
        or head._parameters.get("bias") is not None
        or isinstance(pred, DTensor)
        or spmd.is_type_checking()
    ):
        return None
    weight = head._parameters.get("weight")
    if (
        not isinstance(weight, DTensor)
        or weight.device_mesh.ndim != 1
        or weight.placements != (Shard(0),)
        or weight.dtype != torch.float32
        or not weight.requires_grad
        or isinstance(weight._local_tensor, _ShardedFSDPTensor)
    ):
        return None

    # SimpleFSDP keeps its precision policy in the weight getter closure.
    weight_property = getattr(type(head), "weight", None)
    if not isinstance(weight_property, property):
        return None
    defaults = getattr(weight_property.fget, "__defaults__", None)
    if not defaults or not isinstance(defaults[-1], ReplicateComputation):
        return None
    parametrization = defaults[-1]
    reduce_dtype = parametrization.reduce_dtype or parametrization.param_dtype or weight.dtype
    if reduce_dtype != torch.float32 or parametrization.non_dp_replicate_axes:
        return None
    return parametrization


class CoalescedChunkedLoss(ChunkedLossWrapperWithParamGrads):
    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapperWithParamGrads.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._coalesced_rs_logged = False

    def __call__(self, pred, labels, global_loss_token_counts=None, **loss_inputs):
        head = self.lm_head
        if not isinstance(pred, torch.Tensor) or not pred.requires_grad or self.num_chunks <= 1 or head is None:
            return super().__call__(pred, labels, global_loss_token_counts, **loss_inputs)

        parametrization = _coalescible_parametrization(head, pred)
        if not self._coalesced_rs_logged:
            if parametrization is not None:
                logger.info("[PATCH] GraphTrainer chunked loss: coalescing SimpleFSDP gradients before reduce-scatter")
            else:
                logger.warning(
                    "[PATCH] Chunk-loss RS coalescing is unsupported for this head/layout; using upstream loss"
                )
            self._coalesced_rs_logged = True
        if parametrization is None:
            return super().__call__(pred, labels, global_loss_token_counts, **loss_inputs)

        # Keep the upstream all-gather and compute dtype and defer gradient sync.
        weight = head.weight
        sharded_weight = head._parameters["weight"]
        assert isinstance(weight, torch.Tensor)
        assert isinstance(sharded_weight, DTensor)
        local_head = _LocalGradientLMHead(weight, sharded_weight, parametrization)
        self.lm_head = local_head
        try:
            return super().__call__(pred, labels, global_loss_token_counts, **loss_inputs)
        finally:
            local_head.grad_hook.remove()
            self.lm_head = head

    def _gradient_backprop(self, hidden_states, accumulated_grads, total_loss):
        lm_head = self.lm_head
        if not isinstance(lm_head, _LocalGradientLMHead):
            return super()._gradient_backprop(hidden_states, accumulated_grads, total_loss)
        assert len(hidden_states) == len(accumulated_grads) == 1
        assert lm_head.accumulated_grad is not None
        return _ChunkGradsBackprop.apply(
            hidden_states[0],
            lm_head.sharded_weight,
            accumulated_grads[0],
            lm_head.accumulated_grad,
            total_loss,
            lm_head.parametrization,
        )
