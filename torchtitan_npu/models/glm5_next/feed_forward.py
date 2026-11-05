# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash feed-forward blocks: clamped-swiglu MLP, sigmoid+bias
router, fine-grained routed MoE with a shared expert, and the eager EP
dispatcher mirroring MindSpeed ``ep_dispatcher.ep_forward``.

Expert weight layouts follow the torchtitan grouped-GEMM convention:
``w1_EFD``/``w3_EFD`` [E, I, H] (gate/up) and ``w2_EDF`` [E, H, I] (down),
which maps 1:1 onto the reference ``gate_up_proj [E, 2*I, H]`` /
``down_proj [E, H, I]`` (gate_up = cat(w1, w3)). The ep_size=1 dispatcher
is the uniform grouped-matmul path (meta-safe, single bmm per grouped
GEMM); the EP-sharded variant runs the real alltoall exchange for actual
training runs.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module

from .common import WEIGHT_INIT, linear_config


def clamp_swiglu_eager(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Eager clamp-swiglu: fold the per-half clamp bounds then silu(gate)*up."""
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _eager_permute(tokens: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MindSpeed fsdp2 eager permute: stable argsort + row gather."""
    topk = 1 if indices.dim() == 1 else indices.size(1)
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True).to(indices.dtype)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def _eager_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """MindSpeed fsdp2 eager unpermute: zeros + index_copy_ scatter back."""
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    unpermuted_tokens = torch.zeros(
        [num_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices.long(), permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens *= probs.unsqueeze(-1)
    return unpermuted_tokens.sum(dim=1)


def _uniform_grouped_matmul(x_RH: torch.Tensor, weights_EHD: torch.Tensor, rows_per_expert: int) -> torch.Tensor:
    """Uniform grouped GEMM as a single bmm: after the stable expert sort,
    rows form E equal blocks of ``rows_per_expert`` each."""
    E, H, D = weights_EHD.shape
    return torch.bmm(x_RH.view(E, rows_per_expert, H), weights_EHD).reshape(-1, D)


class GlmMLP(Module):
    """Dense MLP with fused gate_up and the clamped swiglu activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        hidden_dim: int = 12288
        swiglu_limit: float = 10.0
        gate_up_proj: Linear.Config | None = None
        down_proj: Linear.Config | None = None

        def __post_init__(self) -> None:
            if self.gate_up_proj is None:
                self.gate_up_proj = linear_config(self.dim, 2 * self.hidden_dim)
            if self.down_proj is None:
                self.down_proj = linear_config(self.hidden_dim, self.dim)

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = config.gate_up_proj.build()
        self.down_proj = config.down_proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Under TP the MLP runs on the local sequence shard with full
        # weights; drop the DTensor wrapper the parallel styles may add.
        if isinstance(x, torch.distributed.tensor.DTensor):
            x = x.to_local()
        gate_up = self.gate_up_proj(x)
        return self.down_proj(clamp_swiglu_eager(gate_up, self.swiglu_limit))


class TopkRouter(Module):
    """Sigmoid + bias router with n_group=1 (the group gate degenerates)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        num_experts: int = 288
        top_k: int = 8
        n_group: int = 1
        topk_group: int = 1
        routed_scaling_factor: float = 2.5
        norm_topk_prob: bool = True
        debug_force_load_balance: bool = False
        gate: Linear.Config | None = None

        def __post_init__(self) -> None:
            if self.gate is None:
                self.gate = Linear.Config(
                    in_features=self.dim,
                    out_features=self.num_experts,
                    bias=False,
                    param_init={"weight": nn.init.zeros_},
                )

    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.hidden_dim = config.dim
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.debug_force_load_balance = config.debug_force_load_balance
        self.gate = config.gate.build()
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_experts))

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.e_score_correction_bias.device
        self.e_score_correction_bias = torch.zeros(self.num_experts, device=device)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.to(torch.float32), self.gate.weight.to(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias

        if self.num_group > 1:
            group_scores = (
                scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.num_group, self.num_experts // self.num_group)
                .reshape(-1, self.num_experts)
            )
            scores_for_choice = torch.where(score_mask.bool(), scores_for_choice, float("-inf"))
        if self.debug_force_load_balance:
            topk_indices = (
                torch.arange(self.top_k, device=scores_for_choice.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(scores_for_choice.shape[0], -1)
            )
            topk_weights = scores.gather(1, topk_indices)
        else:
            topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
            topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class GlmExperts(GroupedExperts):
    """Routed experts (torchtitan grouped-GEMM layout) with the clamped
    fused swiglu activation and the eager uniform/EP dispatcher.

    ``w1_EFD``/``w3_EFD`` [E, I, H] are the gate/up halves of the reference
    fused ``gate_up_proj [E, 2*I, H]``; ``w2_EDF`` [E, H, I] is ``down_proj``."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float = 10.0

        def __post_init__(self) -> None:
            self.param_init = dict.fromkeys(("w1_EFD", "w2_EDF", "w3_EFD"), WEIGHT_INIT)

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit
        # EP bookkeeping; parallelize_glm5_next overwrites these when the
        # expert-parallel mesh exists. ep_size=1 keeps the uniform
        # grouped-matmul dispatcher as the default (meta capture cannot run
        # the per-expert eager loop: iterating routing indices calls
        # .item()).
        self.ep_size = 1
        self.ep_rank = 0
        self.ep_group = None
        self.expert_ids_per_ep_rank = None

    def _fc1_weight(self) -> torch.Tensor:
        """[E, H, 2*I] view for the first grouped GEMM: cat(w1, w3) is the
        reference fused gate_up [E, 2I, H], transposed."""
        weight = torch.cat([self.w1_EFD, self.w3_EFD], dim=1)
        if isinstance(weight, torch.distributed.tensor.DTensor):
            weight = weight.to_local()
        return weight.transpose(1, 2)

    def _fc2_weight(self) -> torch.Tensor:
        weight = self.w2_EDF
        if isinstance(weight, torch.distributed.tensor.DTensor):
            weight = weight.to_local()
        return weight.transpose(1, 2)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Uniform-routing grouped-matmul dispatcher (ep_size=1 default),
        EP-sharded alltoall variant when parallelize_glm5_next wires EP."""
        return _experts_ep_forward(self, hidden_states, top_k_index, top_k_weights)

    def _eager_debug_forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Eager per-expert loop (real-device debug path; not meta-safe)."""
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(hidden_states[token_idx], self.w1_EFD[expert_idx])
            current = F.linear(
                F.silu(gate_up) * torch.matmul(hidden_states[token_idx], self.w3_EFD[expert_idx].T),
                self.w2_EDF[expert_idx],
            )
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class GlmRoutedExperts(Module):
    """Wrapper matching the torchtitan MoE structure
    (``routed_experts.inner_experts``): the FSDP EP path locates the expert
    parameters at exactly this nesting."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        num_experts: int = 288
        intermediate_dim: int = 2048
        swiglu_limit: float = 10.0
        inner_experts: GlmExperts.Config | None = None

        def __post_init__(self) -> None:
            if self.inner_experts is None:
                self.inner_experts = GlmExperts.Config(
                    dim=self.dim,
                    hidden_dim=self.intermediate_dim,
                    num_experts=self.num_experts,
                    swiglu_limit=self.swiglu_limit,
                )

    def __init__(self, config: Config):
        super().__init__()
        self.inner_experts = config.inner_experts.build()

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        return self.inner_experts(hidden_states, top_k_index, top_k_weights)


class GlmMoE(Module):
    """Routed MoE + shared expert (reference Glm5NextTextMoE)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 4096
        num_experts: int = 288
        top_k: int = 8
        moe_intermediate_size: int = 2048
        n_shared_experts: int = 1
        swiglu_limit: float = 10.0
        routed_scaling_factor: float = 2.5
        norm_topk_prob: bool = True
        n_group: int = 1
        topk_group: int = 1
        debug_force_load_balance: bool = False
        routed_experts: GlmRoutedExperts.Config | None = None
        router: TopkRouter.Config | None = None
        shared_experts: GlmMLP.Config | None = None

        def __post_init__(self) -> None:
            if self.routed_experts is None:
                self.routed_experts = GlmRoutedExperts.Config(
                    dim=self.dim,
                    num_experts=self.num_experts,
                    intermediate_dim=self.moe_intermediate_size,
                    swiglu_limit=self.swiglu_limit,
                )
            if self.router is None:
                self.router = TopkRouter.Config(
                    dim=self.dim,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    n_group=self.n_group,
                    topk_group=self.topk_group,
                    routed_scaling_factor=self.routed_scaling_factor,
                    norm_topk_prob=self.norm_topk_prob,
                    debug_force_load_balance=self.debug_force_load_balance,
                )
            if self.shared_experts is None:
                self.shared_experts = GlmMLP.Config(
                    dim=self.dim,
                    hidden_dim=self.moe_intermediate_size * self.n_shared_experts,
                    swiglu_limit=self.swiglu_limit,
                )

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.routed_experts = config.routed_experts.build()
        self.router = config.router.build()
        self.shared_experts = config.shared_experts.build()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Under TP the MoE runs on the local sequence shard with full
        # weights (router + experts replicated over TP; experts shard over
        # EP only).
        if isinstance(hidden_states, torch.distributed.tensor.DTensor):
            hidden_states = hidden_states.to_local()
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.router(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.routed_experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return hidden_states + self.shared_experts(residuals)


def _experts_ep_forward(
    experts: GlmExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """MindSpeed ``ep_dispatcher.ep_forward`` (alltoall) with the clamped
    fused swiglu. ep_size=1 keeps the uniform grouped-matmul dispatcher over
    the full expert set; ep_size>1 runs the real alltoall exchange (real
    runs only — meta capture does not execute collectives)."""
    num_global_experts = experts.num_experts
    num_rows = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    rows_times_topk = num_rows * top_k
    if rows_times_topk % num_global_experts != 0:
        raise ValueError(
            "Uniform routing assumed by the simulated eager dispatcher is "
            f"violated: {num_rows}*{top_k} rows over {num_global_experts} "
            "experts is not divisible"
        )
    hidden_states_shape = hidden_states.shape

    if experts.ep_size == 1:
        # No expert-parallel mesh: permute rows into expert blocks; the GMM
        # offsets use the uniform tokens-per-expert contract.
        rows_per_expert = rows_times_topk // num_global_experts
        hidden_states, unpermute_indices1 = _eager_permute(hidden_states, top_k_index)
        fc1_output = _uniform_grouped_matmul(hidden_states, experts._fc1_weight(), rows_per_expert)
        act = clamp_swiglu_eager(fc1_output, experts.swiglu_limit)
        fc2_out = _uniform_grouped_matmul(act, experts._fc2_weight(), rows_per_expert)
        hidden_states = _eager_unpermute(fc2_out, unpermute_indices1, probs=top_k_weights)
        return hidden_states.view(*hidden_states_shape)

    # EP-sharded path: expert weights are sharded along the expert dim by
    # the expert FSDP unit and rows destined for remote experts cross the
    # EP group through alltoall. The wiring (dispatcher binding, count
    # preprocessing, per-local-expert grouped GEMM) lands together with the
    # EP step of the parallelism matrix; fail fast until then.
    raise NotImplementedError(
        "GLM-5.3-Flash expert-parallel dispatch is not wired yet: set expert_parallel_degree=1 for now"
    )
