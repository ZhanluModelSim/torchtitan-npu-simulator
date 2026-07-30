# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 feed-forward modules: SiTU-GLU activation and Stable LatentMoE.

Fork reason: SiTU-GLU and LatentMoE are novel to Kimi K3, not in upstream torchtitan.
Reference: MindSpeed-MM mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_linear.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module


class SituGLU(nn.Module):
    """SiTU-GLU activation: beta * tanh(gate/beta) * sigmoid(gate) * up.

    When linear_beta is set, up is also transformed:
    up = linear_beta * tanh(up / linear_beta)
    """

    def __init__(self, beta: float = 4.0, linear_beta: float | None = 25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


class KimiBlockSparseMLP(nn.Module):
    """Per-expert MLP with SiTU-GLU activation.

    Weight layout matches MindSpeed-MM's KimiBlockSparseMLP:
    w1 (gate), w3 (up), w2 (down).
    """

    def __init__(self, hidden_size: int, intermediate_size: int, beta: float = 4.0, linear_beta: float | None = 25.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = SituGLU(beta=beta, linear_beta=linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        return self.w2(self.act_fn(gate_up))


class KimiMLP(nn.Module):
    """Standard MLP for shared experts with SiTU-GLU."""

    def __init__(self, hidden_size: int, intermediate_size: int, beta: float = 4.0, linear_beta: float | None = 25.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SituGLU(beta=beta, linear_beta=linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class KimiMoEGate(nn.Module):
    """MoE router with sigmoid scoring + e_score_correction_bias (noaux_tc).

    Supports group topk selection when num_expert_group > 1.
    """

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
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.score_func = score_func
        self.num_expert_groups = num_expert_groups
        self.topk_group = topk_group
        self.renormalize = renormalize

        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.e_score_correction_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_flat = hidden_states.view(-1, h)

        logits = F.linear(hidden_flat.float(), self.weight.float())
        if self.score_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.score_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise ValueError(f"Unsupported score_func: {self.score_func}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Group topk selection
        if self.num_expert_groups > 1 and self.num_expert_groups > self.topk_group:
            n_tokens = bsz * seq_len
            group_scores = (
                scores_for_choice.view(n_tokens, self.num_expert_groups, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(n_tokens, self.num_expert_groups, self.num_experts // self.num_expert_groups)
                .reshape(n_tokens, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        else:
            tmp_scores = scores_for_choice

        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.renormalize:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    """Stable LatentMoE block for Kimi K3.

    Flow: gate → latent compress → experts → latent norm → latent decompress + shared
    """

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

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.load_balance_coeff = 0.0  # No auxiliary loss balancing for K3 (uses bias correction)

        self.use_latent_moe = config.routed_expert_hidden_size is not None
        self.moe_hidden_size = config.routed_expert_hidden_size or config.hidden_size

        # Router
        self.gate = KimiMoEGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            score_func=config.score_func,
            num_expert_groups=config.num_expert_groups,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            renormalize=config.renormalize,
        )

        # Latent projections
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, bias=False)
            if config.latent_moe_use_norm:
                self.routed_expert_norm = nn.RMSNorm(self.moe_hidden_size, eps=config.norm_eps)

        # Routed experts
        self.experts = nn.ModuleList([
            KimiBlockSparseMLP(
                hidden_size=self.moe_hidden_size,
                intermediate_size=config.moe_intermediate_size,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )
            for _ in range(config.num_experts)
        ])

        # Shared experts
        if config.num_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Latent compression
        if self.use_latent_moe:
            hidden_flat = self.routed_expert_down_proj(hidden_flat)

        # Expert computation
        y = self._moe_forward(hidden_flat, topk_idx, topk_weight)

        # Latent decompression
        if self.use_latent_moe:
            if self.config.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        # Add shared experts
        if self.config.num_shared_experts > 0:
            y = y + self.shared_experts(identity)

        return y

    def _moe_forward(
        self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """Standard per-expert loop (non-fused). NPU fused path via converter."""
        # Meta device: use simplified forward (all experts process all tokens equally)
        if x.device.type == "meta":
            return self._moe_forward_meta(x, topk_idx, topk_weight)

        final_hidden_states = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor.item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = x[token_idx]
            expert_out = self.experts[expert_idx](current_state)
            expert_out = expert_out * topk_weight[token_idx, top_k_pos, None].to(expert_out.dtype)
            final_hidden_states.index_add_(0, token_idx, expert_out.to(final_hidden_states.dtype))

        return final_hidden_states

    def _moe_forward_meta(
        self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """Meta-device MoE forward: deterministic round-robin, no data-dependent ops."""
        num_tokens = x.shape[0]
        # Each token goes through top_k experts; use first expert for shape inference
        expert_out = self.experts[0](x)
        # Scale by average routing weight
        scale = topk_weight.sum(dim=-1, keepdim=True) / self.top_k
        return expert_out * scale
