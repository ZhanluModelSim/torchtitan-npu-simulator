# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE compatibility layer for Block Diffusion."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common import moe as common_moe
from torchtitan.protocols.module import Module


def _expert_histogram(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Return fixed-size expert counts on CPU, NPU, fake and meta devices."""

    return torch.histc(
        indices.to(torch.float32).reshape(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )


class BlockDiffusionRouter(common_moe.TokenChoiceTopKRouter):
    """Token-choice router with a PyTorch-2.12-compatible histogram."""

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        autocast = (
            nullcontext()
            if x.device.type in {"cpu", "meta"}
            else torch.autocast(device_type=x.device.type, dtype=torch.float32)
        )
        with autocast:
            scores = self.gate(x)

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(
                scores_for_choice
            )
        _, selected_experts_indices = torch.topk(
            scores_for_choice,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = (
                self._debug_force_load_balance_routing(scores)
            )
        if self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return (
            top_scores,
            selected_experts_indices,
            _expert_histogram(selected_experts_indices, self.num_experts),
        )


class BlockDiffusionTokenReorderer(common_moe.TokenReorderer):
    """Static-shape token reorderer with portable expert counting."""

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens_per_expert = _expert_histogram(
            selected_experts_indices,
            self.num_experts,
        )
        permutation = torch.argsort(selected_experts_indices.reshape(-1), stable=True)
        sorted_scores = top_scores.reshape(-1)[permutation]
        token_indices = permutation // self.top_k
        return sorted_scores, token_indices, num_tokens_per_expert


class BlockDiffusionGroupedExperts(common_moe.GroupedExperts):
    """Grouped experts with a portable eager reference implementation."""

    @dataclass(kw_only=True, slots=True)
    class Config(common_moe.GroupedExperts.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            return super().forward(x, num_tokens_per_expert)

        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        counts = [int(value) for value in num_tokens_per_expert.tolist()]
        x_splits = torch.split(x[: sum(counts)], counts, dim=0)
        outputs = []
        for expert_index, expert_input in enumerate(x_splits):
            gate = F.silu(F.linear(expert_input, w1[expert_index]))
            up = F.linear(expert_input, w3[expert_index])
            outputs.append(F.linear(gate * up, w2[expert_index]))
        return torch.cat(outputs, dim=0)


class BlockDiffusionMoE(common_moe.MoE):
    """Common TorchTitan MoE with portable router/reorderer modules."""

    @dataclass(kw_only=True, slots=True)
    class Config(common_moe.MoE.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.router = BlockDiffusionRouter(config.router)
        self.reorderer = BlockDiffusionTokenReorderer(
            num_experts=config.num_experts,
            top_k=config.router.top_k,
        )

    def verify_module_protocol(self) -> None:
        assert isinstance(self.router, Module)
        assert isinstance(self.reorderer, Module)
