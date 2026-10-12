# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Next feed-forward modules: dense MLP (upstream) and sparse MoE.

The MoE block follows the Qwen3-Next design: softmax routing with top-k and
topk-prob normalization, grouped routed experts, plus a gated shared expert
(``sigmoid(shared_expert_gate(x)) * shared_expert(x)``).

Reference: MindSpeed-LLM
mindspeed_llm/fsdp2/models/qwen3_next/modeling_qwen3_next.py
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


class Qwen3NextGroupedExperts(nn.Module):
    """Routed experts in the ``[E, out, in]`` weight layout.

    Uses ``torch._grouped_mm`` on NPU/meta devices and a per-expert loop
    fallback on CPU. The w1/w2/w3 naming and forward signature match
    torchtitan's ``ExpertParallel`` expectations.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        if x.device.type == "cpu":
            return self._forward_loop(x, num_tokens_per_expert, w1, w2, w3)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate = torch._grouped_mm(
            x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
        )
        up = torch._grouped_mm(
            x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )
        hidden = F.silu(gate) * up
        return torch._grouped_mm(
            hidden,
            w2.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(x)

    def _forward_loop(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor:
        token_counts = num_tokens_per_expert.to(torch.int64).tolist()
        outputs = []
        offset = 0
        for expert_idx, token_count in enumerate(token_counts):
            expert_input = x[offset : offset + token_count]
            gate = F.linear(expert_input, w1[expert_idx])
            up = F.linear(expert_input, w3[expert_idx])
            outputs.append(
                F.linear(
                    F.silu(gate) * up,
                    w2[expert_idx],
                )
            )
            offset += token_count
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)


class Qwen3NextTokenReorderer(nn.Module):
    """Group routed tokens by expert while preserving token-score alignment."""

    def __init__(self, *, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_expert_indices = selected_experts_indices.reshape(-1)
        num_tokens_per_expert = torch.histc(
            flat_expert_indices.float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        ).to(torch.int64)
        sorted_assignment_indices = torch.argsort(
            flat_expert_indices,
            stable=True,
        )
        sorted_scores = top_scores.reshape(-1)[sorted_assignment_indices]
        token_indices = sorted_assignment_indices // self.top_k
        return sorted_scores, token_indices, num_tokens_per_expert


class Qwen3NextMoE(nn.Module):
    """Sparse MoE block with softmax routing and a gated shared expert."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 2048
        num_experts: int = 512
        top_k: int = 10
        moe_intermediate_size: int = 512
        shared_expert_intermediate_size: int = 512
        norm_topk_prob: bool = True
        debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.debug_force_load_balance = config.debug_force_load_balance

        self.gate = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        ).build()
        self.reorderer = Qwen3NextTokenReorderer(
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        self.experts = Qwen3NextGroupedExperts(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )

        self.shared_expert = _SharedExpertMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = Linear.Config(
            in_features=config.hidden_size,
            out_features=1,
            bias=False,
        ).build()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            if hidden_states.device_mesh.ndim != 1:
                raise ValueError(
                    "Qwen3-Next MoE expects a 1D TP DTensor input, "
                    f"got {hidden_states.device_mesh.ndim}D"
                )
            hidden_states = hidden_states.to_local(grad_placements=(Partial(),))

        original_shape = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Router: softmax scores, top-k selection, optional probability norm.
        logits = F.linear(hidden_flat.float(), self.gate.weight.float())
        scores = logits.softmax(dim=-1)
        if self.debug_force_load_balance:
            token_offsets = torch.arange(
                hidden_flat.shape[0], device=hidden_flat.device, dtype=torch.int64
            ) * self.top_k
            expert_offsets = torch.arange(
                self.top_k, device=hidden_flat.device, dtype=torch.int64
            )
            topk_idx = (
                token_offsets.unsqueeze(1) + expert_offsets.unsqueeze(0)
            ) % self.num_experts
        else:
            topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.config.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weight = topk_weight.to(hidden_states.dtype)

        (
            topk_weight,
            token_indices,
            num_tokens_per_expert,
        ) = self.reorderer(topk_weight, topk_idx)
        routed_input = hidden_flat[token_indices]
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = routed_output * topk_weight.unsqueeze(-1).to(
            routed_output.dtype
        )

        combined = torch.zeros_like(hidden_flat)
        combined.index_add_(
            0,
            token_indices,
            routed_output.to(combined.dtype),
        )
        output = combined.view(*original_shape)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        output = output + shared_gate * shared_expert_output
        return output


class _SharedExpertMLP(nn.Module):
    """Dense SwiGLU MLP used as the Qwen3-Next shared expert."""

    def __init__(self, *, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear.Config(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        ).build()
        self.up_proj = Linear.Config(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        ).build()
        self.down_proj = Linear.Config(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=False,
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
