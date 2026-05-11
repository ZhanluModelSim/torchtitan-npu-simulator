# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


def _npu_moe_forward(self, x):
    if isinstance(x, DTensor):
        x = x.to_local(grad_placements=(Partial(),))

    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    # Bypass self.router() entirely.  NoParallel / TP hooks on the gate
    # module convert inputs to DTensors, which then crash when mixed with
    # plain-Tensor expert_bias or NPU kernels.  Instead, compute gate
    # scores directly using local tensors.
    gate = self.router.gate
    gate_weight = gate.weight
    gate_bias = getattr(gate, "bias", None)
    if isinstance(gate_weight, DTensor):
        gate_weight = gate_weight.to_local()
    if gate_bias is not None and isinstance(gate_bias, DTensor):
        gate_bias = gate_bias.to_local()

    with torch.autocast(device_type=x.device.type, dtype=torch.float32):
        scores = torch.nn.functional.linear(x, gate_weight, gate_bias)

    score_func = self.router.score_func
    if score_func == "sigmoid":
        scores = torch.sigmoid(scores)
    elif score_func == "softmax":
        scores = torch.nn.functional.softmax(scores, dim=1)
    else:
        raise NotImplementedError(f"Unknown score function {score_func}")

    expert_bias = self.expert_bias
    scores_for_choice = scores if expert_bias is None else scores + expert_bias

    if self.router.num_expert_groups is not None:
        num_expert_groups = self.router.num_expert_groups
        num_limited_groups = self.router.num_limited_groups
        num_experts = self.router.num_experts
        experts_per_group = num_experts // num_expert_groups
        scores_grouped = scores_for_choice.view(
            -1, num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, num_experts)

    _, selected_experts_indices = torch.topk(
        scores_for_choice, k=self.router.top_k, dim=-1, sorted=False
    )
    top_scores = scores.gather(dim=1, index=selected_experts_indices)

    if self.router.route_norm:
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator
    top_scores = top_scores * self.router.route_scale

    num_tokens_per_expert = torch.histc(
        selected_experts_indices.view(-1),
        bins=self.router.num_experts,
        min=0,
        max=self.router.num_experts,
    )

    if self.shared_experts is not None:
        out = self.shared_experts(x)
        if isinstance(out, DTensor):
            out = out.to_local()
    else:
        out = torch.zeros_like(x)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        # Mixing FP32 `topk_score` and BF16 `routed_output` causes
        # MoeTokenUnpermuteGrad to return NaN values. Cast the FP32
        # part to BF16 as a temporary workaround.
        top_scores.to(x.dtype),
    )
    return (out + unpermuted).reshape(bs, slen, dim)


@register_npu_converter("npu_permute")
class PermuteKernel(BaseConverter):

    MOE_PACKAGE = "torchtitan.models.common.moe"

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        pkg = cls.MOE_PACKAGE

        count = replace_methods("MoE", "forward", _npu_moe_forward, package=pkg)

        # pyrefly: ignore [bad-return]
        return count
