# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""glm5_next feed-forward: clamp-swiglu dense MLP and sigmoid noaux_tc MoE.

Contract: see MODEL_CONTRACT.md (section 5). Experts keep torchtitan's
``[E, out, in]`` grouped layout (w1=gate, w3=up, w2=down) so the upstream
ExpertParallel plan and ``npu_gmm`` w13 fusion conventions apply unchanged.
Routing is sigmoid scoring + ``e_score_correction_bias`` (noaux_tc) with
``n_group=1`` (no group filtering), renormalized top-k weights scaled by
``routed_scaling_factor``.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.linear import Linear

if TYPE_CHECKING:
    from .model import Glm5NextTextModel


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _linear_local(linear, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    return F.linear(x, _local(linear.weight))


def clamp_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """GLM clamp-swiglu: silu(clamp(gate, max=limit)) * clamp(up, ±limit)."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class GlmMLP(nn.Module):
    """Dense clamp-swiglu MLP.

    When ``_tp_group`` is set (shared-expert inter-dim TP), the forward runs
    on local weights and all-reduces the rowwise output so the Partial sum
    never leaks into plain-tensor ops.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self._tp_group = None
        self.gate_proj = Linear.Config(in_features=hidden_size, out_features=intermediate_size, bias=False).build()
        self.up_proj = Linear.Config(in_features=hidden_size, out_features=intermediate_size, bias=False).build()
        self.down_proj = Linear.Config(in_features=intermediate_size, out_features=hidden_size, bias=False).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_group is not None:
            import torch.distributed.nn.functional as dist_nn

            gate = _linear_local(self.gate_proj, x)
            up = _linear_local(self.up_proj, x)
            hidden = clamp_swiglu(gate, up, self.swiglu_limit)
            out = _linear_local(self.down_proj, hidden)
            return dist_nn.all_reduce(out, group=self._tp_group)
        return self.down_proj(clamp_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """Attach MoE auxiliary-loss gradients without changing forward values."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):  # noqa: ANN001
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: ANN001
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_grad = torch.ones_like(aux_loss) * MoEAuxLossAutoScaler.main_loss_backward_scale
        return grad_output, aux_loss_backward_grad

    @classmethod
    def set_loss_scale(cls, scale: torch.Tensor) -> None:
        cls.main_loss_backward_scale = scale


def compute_load_balance_loss(
    router_probs: torch.Tensor, router_indices: torch.Tensor, num_experts: int, topk: int
) -> torch.Tensor:
    with torch.no_grad():
        one_hot = torch.zeros(
            router_indices.shape[0],
            topk,
            num_experts,
            dtype=router_probs.dtype,
            device=router_indices.device,
        )
        one_hot.scatter_(2, router_indices.unsqueeze(-1), 1.0)
        density = one_hot.mean(dim=(0, 1))
    density_proxy = router_probs.mean(dim=0)
    return (density * density_proxy).sum() * num_experts


def _round_robin_topk(num_tokens: int, top_k: int, num_experts: int, device: torch.device) -> torch.Tensor:
    token_offsets = torch.arange(num_tokens, device=device, dtype=torch.int64) * top_k
    expert_offsets = torch.arange(top_k, device=device, dtype=torch.int64)
    return (token_offsets.unsqueeze(1) + expert_offsets.unsqueeze(0)) % num_experts


class GlmMoEGate(nn.Module):
    """Sigmoid router with correction bias (noaux_tc, n_group=1)."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
        debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.renormalize = renormalize
        self.debug_force_load_balance = debug_force_load_balance
        self.gate = Linear.Config(in_features=hidden_size, out_features=num_experts, bias=False).build()
        # noaux_tc correction bias: updated by the MoE load-balancing hook
        # (heuristic), not by gradients (topk over it is non-differentiable).
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(tokens.float(), self.gate.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + _local(self.e_score_correction_bias).float()

        if self.debug_force_load_balance:
            topk_indices = _round_robin_topk(tokens.shape[0], self.top_k, self.num_experts, tokens.device)
        else:
            topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.renormalize and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        aux_loss = compute_load_balance_loss(scores, topk_indices, self.num_experts, self.top_k)
        return topk_indices, topk_weights, aux_loss


class GlmGroupedExperts(nn.Module):
    """Routed experts with ``[E, out, in]`` grouped weights (w1/w3/w2)."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        if x.device.type == "cpu":
            return self._forward_loop(x, num_tokens_per_expert, w1, w2, w3)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate = torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
        up = torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
        hidden = clamp_swiglu(gate, up, self.swiglu_limit)
        return torch._grouped_mm(hidden, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

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
            offset += token_count
            if expert_input.shape[0] == 0:
                continue
            gate = F.linear(expert_input, w1[expert_idx])
            up = F.linear(expert_input, w3[expert_idx])
            outputs.append(F.linear(clamp_swiglu(gate, up, self.swiglu_limit), w2[expert_idx]))
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)


class GlmTokenReorderer(nn.Module):
    """Group routed tokens by expert while preserving token-score alignment."""

    def __init__(self, num_experts: int, top_k: int):
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
        sorted_assignment_indices = torch.argsort(flat_expert_indices, stable=True)
        sorted_scores = top_scores.reshape(-1)[sorted_assignment_indices]
        token_indices = sorted_assignment_indices // self.top_k
        return sorted_scores, token_indices, num_tokens_per_expert


class GlmSparseMoeBlock(nn.Module):
    """2048-expert MoE block: sigmoid noaux_tc router + shared expert."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_shared_experts: int,
        moe_intermediate_size: int,
        routed_scaling_factor: float,
        swiglu_limit: float,
        debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        # Consumed by the MoE load-balancing optimizer hook
        # (torchtitan_npu/patches/torchtitan/optimizer.py).
        self.load_balance_coeff = 1e-3
        self.gate = GlmMoEGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=True,
            debug_force_load_balance=debug_force_load_balance,
        )
        self.reorderer = GlmTokenReorderer(num_experts, num_experts_per_tok)
        self.experts = GlmGroupedExperts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            swiglu_limit=swiglu_limit,
        )
        if num_shared_experts > 0:
            self.shared_experts = GlmMLP(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size * num_shared_experts,
                swiglu_limit=swiglu_limit,
            )
        else:
            self.shared_experts = None
        # Routing histogram consumed by the MoE load-balancing optimizer hook
        # (torchtitan_npu/patches/torchtitan/optimizer.py).
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.int64),
            persistent=False,
        )
        # noaux_tc: ``e_score_correction_bias`` stays a fixed zero-init
        # parameter in v1 (no gradient, no heuristic hook update). The
        # simulator additionally forces round-robin routing for
        # deterministic GMM shapes (MODEL_CONTRACT.md section 5).

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            if hidden_states.device_mesh.ndim != 1:
                raise ValueError(
                    f"glm5_next MoE expects a 1D TP DTensor input, got {hidden_states.device_mesh.ndim}D"
                )
            hidden_states = hidden_states.to_local(grad_placements=(Partial(),))

        identity = hidden_states
        original_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden_states.shape[-1])

        topk_idx, topk_weight, aux_loss = self.gate(tokens)
        sorted_scores, token_indices, num_tokens_per_expert = self.reorderer(topk_weight, topk_idx)
        with torch.no_grad():
            self.tokens_per_expert.copy_(num_tokens_per_expert)
        routed_input = tokens[token_indices]
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = routed_output * sorted_scores.unsqueeze(-1).to(routed_output.dtype)
        routed_output = MoEAuxLossAutoScaler.apply(routed_output, aux_loss)

        combined = torch.zeros_like(tokens)
        combined.index_add_(0, token_indices, routed_output.to(combined.dtype))
        output = combined.view(original_shape)
        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output


def estimate_expert_params(hidden_size: int, moe_intermediate_size: int) -> int:
    return 2 * moe_intermediate_size * hidden_size + hidden_size * moe_intermediate_size
