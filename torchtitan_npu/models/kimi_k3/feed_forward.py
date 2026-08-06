# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 feed-forward modules: SiTU-GLU and Stable LatentMoE."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.module import Module


class SituGLU(nn.Module):
    """SiTU-GLU activation used by both dense and routed experts."""

    def __init__(self, beta: float = 4.0, linear_beta: float | None = 25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate_float = gate.float()
        up_float = up.float()
        activated_gate = (
            self.beta
            * torch.tanh(gate_float / self.beta)
            * torch.sigmoid(gate_float)
        )
        if self.linear_beta is not None:
            up_float = self.linear_beta * torch.tanh(
                up_float / self.linear_beta
            )
        return (activated_gate * up_float).to(x.dtype)


class KimiMLP(Module):
    """Dense/shared Kimi MLP with SiTU-GLU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int
        beta: float = 4.0
        linear_beta: float | None = 25.0

    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SituGLU(
            beta=config.beta,
            linear_beta=config.linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class KimiGroupedExperts(nn.Module):
    """Routed experts in torchtitan's ``[E, out, in]`` weight layout."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        beta: float,
        linear_beta: float | None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.w3 = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.act_fn = SituGLU(beta=beta, linear_beta=linear_beta)

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
            x.bfloat16(),
            w1.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        up = torch._grouped_mm(
            x.bfloat16(),
            w3.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        hidden = self.act_fn(torch.cat((gate, up), dim=-1))
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
                    self.act_fn(torch.cat((gate, up), dim=-1)),
                    w2[expert_idx],
                )
            )
            offset += token_count
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)


class KimiRouterLinear(nn.Linear):
    """Router projection evaluated in float32, matching the reference model."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input_tensor.float(),
            self.weight.float(),
            None,
        )


class KimiMoEGate(nn.Module):
    """Kimi sigmoid router with correction bias and optional group limiting."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_func: str = "sigmoid",
        num_expert_groups: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
        debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.score_func = score_func
        self.num_expert_groups = num_expert_groups
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.debug_force_load_balance = debug_force_load_balance

        self.gate = KimiRouterLinear(
            hidden_size,
            num_experts,
            bias=False,
        )
        self.register_parameter(
            "e_score_correction_bias",
            nn.Parameter(torch.zeros(num_experts)),
        )
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        logits = self.gate(hidden_flat)
        if self.score_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.score_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise ValueError(f"Unsupported score_func: {self.score_func}")

        scores_for_choice = (
            scores + self.e_score_correction_bias.unsqueeze(0)
        )
        if self.num_expert_groups > 1:
            if self.num_experts % self.num_expert_groups != 0:
                raise ValueError(
                    "num_experts must be divisible by num_expert_groups"
                )
            if self.topk_group >= self.num_expert_groups:
                raise ValueError(
                    "topk_group must be smaller than num_expert_groups"
                )
            num_tokens = hidden_flat.shape[0]
            grouped_scores = scores_for_choice.view(
                num_tokens, self.num_expert_groups, -1
            )
            if grouped_scores.shape[-1] < 2:
                raise ValueError("Each expert group must contain at least 2 experts")
            group_scores = grouped_scores.topk(2, dim=-1)[0].sum(dim=-1)
            group_indices = torch.topk(
                group_scores,
                k=self.topk_group,
                dim=-1,
                sorted=False,
            )[1]
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, group_indices, True)
            scores_for_choice = grouped_scores.masked_fill(
                ~group_mask.unsqueeze(-1),
                float("-inf"),
            ).view(num_tokens, -1)

        if self.debug_force_load_balance:
            token_offsets = (
                torch.arange(
                    hidden_flat.shape[0],
                    device=hidden_flat.device,
                    dtype=torch.int64,
                )
                * self.top_k
            )
            expert_offsets = torch.arange(
                self.top_k,
                device=hidden_flat.device,
                dtype=torch.int64,
            )
            topk_idx = (
                token_offsets.unsqueeze(1) + expert_offsets.unsqueeze(0)
            ) % self.num_experts
        else:
            topk_idx = torch.topk(
                scores_for_choice,
                k=self.top_k,
                dim=-1,
                sorted=False,
            )[1]
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.renormalize:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_idx, topk_weight * self.routed_scaling_factor


class KimiTokenReorderer(nn.Module):
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


class KimiSparseMoeBlock(nn.Module):
    """Stable LatentMoE with grouped routed experts."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 7168
        num_experts: int = 896
        top_k: int = 16
        moe_intermediate_size: int = 3072
        num_shared_experts: int = 2
        routed_expert_hidden_size: int | None = 3584
        latent_moe_use_norm: bool = True
        score_func: str = "sigmoid"
        num_expert_groups: int = 1
        topk_group: int = 1
        routed_scaling_factor: float = 1.0
        renormalize: bool = True
        situ_beta: float = 4.0
        situ_linear_beta: float | None = 25.0
        norm_eps: float = 1e-5
        debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.load_balance_coeff = None

        self.use_latent_moe = config.routed_expert_hidden_size is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size or config.hidden_size
        )
        self.gate = KimiMoEGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            score_func=config.score_func,
            num_expert_groups=config.num_expert_groups,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            renormalize=config.renormalize,
            debug_force_load_balance=config.debug_force_load_balance,
        )

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
            )
            if config.latent_moe_use_norm:
                self.routed_expert_norm = RMSNorm.Config(
                    normalized_shape=self.moe_hidden_size,
                    eps=config.norm_eps,
                ).build()

        self.reorderer = KimiTokenReorderer(
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        self.experts = KimiGroupedExperts(
            num_experts=config.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
        )

        if config.num_shared_experts > 0:
            self.shared_experts = KimiMLP.Config(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.num_shared_experts
                ),
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            ).build()
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            if hidden_states.device_mesh.ndim != 1:
                raise ValueError(
                    "Kimi MoE expects a 1D TP DTensor input, "
                    f"got {hidden_states.device_mesh.ndim}D"
                )
            hidden_states = hidden_states.to_local(grad_placements=(Partial(),))

        identity = hidden_states
        original_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_flat = self.routed_expert_down_proj(hidden_flat)

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

        if self.use_latent_moe:
            if self.config.latent_moe_use_norm:
                combined = self.routed_expert_norm(combined)
            combined = self.routed_expert_up_proj(combined)

        output = combined.view(*original_shape)
        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output
