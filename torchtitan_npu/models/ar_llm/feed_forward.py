# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ar_llm LatentMoE: token/expert-choice routers and 5-matrix latent experts.

Expert structure (routed and shared share the same shape family):
    d -> latent (gate_down / up_down)
    latent -> inter (latent_to_inter, shared for gate and up)
    SwiGLU(clamp) at inter
    inter -> latent (inter_to_latent)
    latent -> d (latent_to_out)

Routed experts keep the torchtitan ``[E, out, in]`` grouped weight layout so
``torch._grouped_mm`` and the upstream ExpertParallel plan can be reused.
Weight naming: ``w1``=gate_down, ``w3``=up_down, ``w4``=latent_to_inter,
``w5``=inter_to_latent, ``w2``=latent_to_out.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm

from .attention import swiglu

if TYPE_CHECKING:
    from .model import ArLlmModel


def sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.softplus(x))


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


class LatentExpertMLP(nn.Module):
    """Dense latent expert (used for the shared-expert group)."""

    _tp_group = None

    def __init__(self, model_args: "ArLlmModel.Config", num_experts: int = 1):
        super().__init__()
        d = model_args.dim
        latent = model_args.moe_latent_dim
        inter = model_args.moe_intermediate_size
        self.num_experts = num_experts
        self.clamp_val = model_args.swiglu_clamp
        self.gate_down = nn.Parameter(torch.empty(num_experts, latent, d))
        self.up_down = nn.Parameter(torch.empty(num_experts, latent, d))
        self.latent_to_inter = nn.Parameter(torch.empty(num_experts, inter, latent))
        self.inter_to_latent = nn.Parameter(torch.empty(num_experts, latent, inter))
        self.latent_to_out = nn.Parameter(torch.empty(num_experts, d, latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_weights(
            x,
            self._local(self.gate_down),
            self._local(self.up_down),
            self._local(self.latent_to_inter),
            self._local(self.inter_to_latent),
            self._local(self.latent_to_out),
        )
        return out.sum(dim=0) if self.num_experts > 1 else out.squeeze(0)

    @staticmethod
    def _local(w: torch.Tensor) -> torch.Tensor:
        return w.to_local() if isinstance(w, DTensor) else w

    def _forward_weights(
        self,
        x: torch.Tensor,
        gate_down: torch.Tensor,
        up_down: torch.Tensor,
        latent_to_inter: torch.Tensor,
        inter_to_latent: torch.Tensor,
        latent_to_out: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.einsum("bsd,nld->nbsl", x, gate_down)
        up = torch.einsum("bsd,nld->nbsl", x, up_down)
        gate_inter = torch.einsum("nbsl,nil->nbsi", gate, latent_to_inter)
        up_inter = torch.einsum("nbsl,nil->nbsi", up, latent_to_inter)
        act = swiglu(up_inter, gate_inter, self.clamp_val)
        latent_out = torch.einsum("nbsi,nli->nbsl", act, inter_to_latent)
        if self._tp_group is not None:
            import torch.distributed.nn.functional as dist_nn

            latent_out = dist_nn.all_reduce(latent_out, group=self._tp_group)
        return torch.einsum("nbsl,ndl->nbsd", latent_out, latent_to_out)


class LatentGroupedExperts(nn.Module):
    """Routed latent experts with ``[E, out, in]`` grouped weights."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        d = model_args.dim
        latent = model_args.moe_latent_dim
        inter = model_args.moe_intermediate_size
        e = model_args.num_routed_experts
        self.num_experts = e
        self.clamp_val = model_args.swiglu_clamp
        self._tp_group = None
        self.w1 = nn.Parameter(torch.empty(e, latent, d))
        self.w3 = nn.Parameter(torch.empty(e, latent, d))
        self.w4 = nn.Parameter(torch.empty(e, inter, latent))
        self.w5 = nn.Parameter(torch.empty(e, latent, inter))
        self.w2 = nn.Parameter(torch.empty(e, d, latent))

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        w1 = self._local(self.w1)
        w2 = self._local(self.w2)
        w3 = self._local(self.w3)
        w4 = self._local(self.w4)
        w5 = self._local(self.w5)

        if x.device.type == "cpu":
            return self._forward_loop(x, num_tokens_per_expert, w1, w2, w3, w4, w5)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate = torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
        up = torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
        gate_inter = torch._grouped_mm(gate, w4.bfloat16().transpose(-2, -1), offs=offsets)
        up_inter = torch._grouped_mm(up, w4.bfloat16().transpose(-2, -1), offs=offsets)
        act = swiglu(up_inter, gate_inter, self.clamp_val)
        latent_out = torch._grouped_mm(act, w5.bfloat16().transpose(-2, -1), offs=offsets)
        if self._tp_group is not None:
            import torch.distributed.nn.functional as dist_nn

            latent_out = dist_nn.all_reduce(latent_out.to(x.dtype), group=self._tp_group)
        return torch._grouped_mm(latent_out.bfloat16(), w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    @staticmethod
    def _local(w: torch.Tensor) -> torch.Tensor:
        return w.to_local() if isinstance(w, DTensor) else w

    def _forward_loop(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        w4: torch.Tensor,
        w5: torch.Tensor,
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
            gate_inter = F.linear(gate, w4[expert_idx])
            up_inter = F.linear(up, w4[expert_idx])
            act = swiglu(up_inter, gate_inter, self.clamp_val)
            latent_out = F.linear(act, w5[expert_idx])
            if self._tp_group is not None:
                import torch.distributed.nn.functional as dist_nn

                latent_out = dist_nn.all_reduce(latent_out, group=self._tp_group)
            outputs.append(F.linear(latent_out, w2[expert_idx]))
        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(x)


def _round_robin_topk(num_tokens: int, top_k: int, num_experts: int, device: torch.device) -> torch.Tensor:
    token_offsets = torch.arange(num_tokens, device=device, dtype=torch.int64) * top_k
    expert_offsets = torch.arange(top_k, device=device, dtype=torch.int64)
    return (token_offsets.unsqueeze(1) + expert_offsets.unsqueeze(0)) % num_experts


class ArLlmTokenRouter(nn.Module):
    """Token-choice router with sqrtsoftplus scoring and round-robin debug mode."""

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        self.num_experts = model_args.num_routed_experts
        self.top_k = model_args.num_experts_per_token
        self.route_scale = model_args.route_scale
        self.score_func = model_args.router_score_function
        self.debug_force_load_balance = model_args.debug_force_load_balance
        self.gate = Linear.Config(
            in_features=model_args.dim, out_features=model_args.num_routed_experts, bias=False
        ).build()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(tokens.float(), self.gate.weight.float())
        if self.score_func == "sqrtsoftplus":
            scores = sqrtsoftplus(logits) * self.route_scale
        elif self.score_func == "softmax":
            scores = logits.softmax(dim=-1) * self.route_scale
        elif self.score_func == "sigmoid":
            scores = logits.sigmoid() * self.route_scale
        else:
            raise ValueError(f"Unsupported router_score_function: {self.score_func}")

        if self.debug_force_load_balance:
            topk_idx = _round_robin_topk(tokens.shape[0], self.top_k, self.num_experts, tokens.device)
        else:
            topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        aux_loss = compute_load_balance_loss(logits.softmax(dim=-1), topk_idx, self.num_experts, self.top_k)
        return topk_idx, topk_weight, aux_loss


class ArLlmExpertChoiceRouter(nn.Module):
    """Expert-choice router: each expert selects ``capacity`` tokens.

    The per-token mapping is rebuilt through a [T, E] score matrix so tokens
    selected by more than ``top_k`` experts keep their best selections, and
    tokens selected by fewer experts emit zero-weight rows (raw reference
    semantics). ``num_tokens_per_expert`` therefore stays shape-static.
    """

    def __init__(self, model_args: "ArLlmModel.Config"):
        super().__init__()
        self.num_experts = model_args.num_routed_experts
        self.top_k = model_args.num_experts_per_token
        self.capacity_factor = model_args.mor_expert_capacity
        self.route_scale = model_args.route_scale
        self.debug_force_load_balance = model_args.debug_force_load_balance
        self.gate = Linear.Config(
            in_features=model_args.dim, out_features=model_args.num_routed_experts, bias=False
        ).build()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = tokens.shape[0]
        if self.debug_force_load_balance:
            topk_idx = _round_robin_topk(num_tokens, self.top_k, self.num_experts, tokens.device)
            logits = F.linear(tokens.float(), self.gate.weight.float())
            scores = sqrtsoftplus(logits) * self.route_scale
            topk_weight = scores.gather(1, topk_idx)
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            aux_loss = compute_load_balance_loss(logits.softmax(dim=-1), topk_idx, self.num_experts, self.top_k)
            return topk_idx, topk_weight, aux_loss

        logits = F.linear(tokens.float(), self.gate.weight.float())
        scores = sqrtsoftplus(logits) * self.route_scale

        capacity = max(1, int(self.capacity_factor * num_tokens * self.top_k / self.num_experts))
        capacity = min(capacity, num_tokens)
        selected_scores, selected_tokens = torch.topk(scores.transpose(0, 1), capacity, dim=-1)

        pair_scores = selected_scores.reshape(-1)
        pair_tokens = selected_tokens.reshape(-1)
        pair_experts = (
            torch.arange(self.num_experts, device=tokens.device)
            .unsqueeze(1)
            .expand(-1, capacity)
            .reshape(-1)
        )

        token_expert_scores = torch.full(
            (num_tokens * self.num_experts,),
            float("-inf"),
            device=tokens.device,
            dtype=pair_scores.dtype,
        )
        flat_index = pair_tokens * self.num_experts + pair_experts
        token_expert_scores.scatter_(0, flat_index, pair_scores)
        token_expert_scores = token_expert_scores.view(num_tokens, self.num_experts)
        topk_scores, topk_experts = torch.topk(token_expert_scores, k=self.top_k, dim=-1)
        topk_weight = topk_scores.clamp(min=0.0)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        with torch.no_grad():
            token_selected_count = torch.zeros(
                num_tokens, device=tokens.device, dtype=torch.float
            )
            token_selected_count.scatter_add_(
                0,
                pair_tokens,
                torch.ones(pair_tokens.numel(), device=tokens.device, dtype=torch.float),
            )
        sampling_loss = token_selected_count.var() / torch.clamp(
            token_selected_count.mean() + 1e-8, min=1.0
        )
        return topk_experts, topk_weight, sampling_loss


class ArLlmTokenReorderer(nn.Module):
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


class ArLlmMoE(nn.Module):
    """LatentMoE block: shared experts + routed experts (token/expert choice)."""

    def __init__(self, model_args: "ArLlmModel.Config", mor_type: str):
        super().__init__()
        self.mor_type = mor_type
        self.num_experts = model_args.num_routed_experts
        self.top_k = model_args.num_experts_per_token
        self.load_balance_coeff = 1e-3
        if mor_type == "expert":
            self.router = ArLlmExpertChoiceRouter(model_args)
        else:
            self.router = ArLlmTokenRouter(model_args)
        self.reorderer = ArLlmTokenReorderer(model_args.num_routed_experts, model_args.num_experts_per_token)
        self.experts = LatentGroupedExperts(model_args)
        if model_args.num_shared_experts > 0:
            self.shared_experts = LatentExpertMLP(model_args, num_experts=model_args.num_shared_experts)
        else:
            self.shared_experts = None
        # Routing histogram consumed by the MoE load-balancing optimizer hook
        # (torchtitan_npu/patches/torchtitan/optimizer.py).
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(model_args.num_routed_experts, dtype=torch.int64),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            if hidden_states.device_mesh.ndim != 1:
                raise ValueError(
                    f"ar_llm MoE expects a 1D TP DTensor input, got {hidden_states.device_mesh.ndim}D"
                )
            hidden_states = hidden_states.to_local(grad_placements=(Partial(),))

        identity = hidden_states
        original_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden_states.shape[-1])

        topk_idx, topk_weight, aux_loss = self.router(tokens)
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


def estimate_expert_params(dim: int, latent: int, inter: int) -> int:
    return 3 * dim * latent + 2 * latent * inter
