# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""mm_gc feed-forward modules: dense SwiGLU MLP and Multi-Head MoE.

The Multi-Head MoE follows the reference deployment semantics: per-head
sigmoid routing with aux-loss-free ``expert_bias`` (selection only),
L1 ``route_norm`` over the gathered top-k scores, ``route_scale`` on the
normalized probabilities, and a head-major CSR sort into a flat
``heads * experts_per_head`` grouped-expert bank so that EP can shard along
the head dimension and GMM can consume the CSR layout directly.
"""

from dataclasses import dataclass

import torch
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


def _local_weight(w: torch.Tensor) -> torch.Tensor:
    return w.to_local() if isinstance(w, DTensor) else w


class MMGcMLP(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int

    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,
        ).build()
        self.up_proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=False,
        ).build()
        self.down_proj = Linear.Config(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadMoEGate(nn.Module):
    """Per-head router: [heads, head_hidden, experts_per_head] weights."""

    def __init__(
        self,
        *,
        moe_num_heads: int,
        head_hidden_size: int,
        experts_per_head: int,
        top_k: int,
        score_func: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 1.0,
    ):
        super().__init__()
        if top_k > experts_per_head:
            raise ValueError(
                f"top_k {top_k} must not exceed experts_per_head {experts_per_head}"
            )
        if score_func not in ("sigmoid", "softmax"):
            raise ValueError(f"Unsupported score_func: {score_func}")
        self.moe_num_heads = moe_num_heads
        self.experts_per_head = experts_per_head
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

        self.router = nn.Parameter(
            torch.empty(moe_num_heads, head_hidden_size, experts_per_head)
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(moe_num_heads * experts_per_head, dtype=torch.float32),
            persistent=False,
        )
        self.local_head_start = 0
        self.local_num_heads = moe_num_heads

    def forward(
        self, x_heads: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router = _local_weight(self.router)
        logits = torch.einsum("nsh,she->nse", x_heads.float(), router.float())
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)

        local_bias = self.expert_bias.view(-1, self.experts_per_head)
        local_bias = local_bias[
            self.local_head_start : self.local_head_start + self.local_num_heads
        ].reshape(-1)
        topk_scores = scores + local_bias.view(
            1, self.local_num_heads, self.experts_per_head
        )
        topk_indices = torch.topk(
            topk_scores, k=self.top_k, dim=-1, sorted=False
        ).indices
        topk_probs = scores.gather(-1, topk_indices)
        if self.route_norm:
            topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=1e-12)
        return topk_probs * self.route_scale, topk_indices


class MMGcGroupedExperts(nn.Module):
    """Flat routed-expert bank in torchtitan's ``[E, out, in]`` layout.

    The bank spans every head's experts: expert ``head * experts_per_head + e``
    operates on head ``head``'s subspace only.
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
            x.bfloat16(),
            w1.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        up = torch._grouped_mm(
            x.bfloat16(),
            w3.bfloat16().transpose(-2, -1),
            offs=offsets,
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


class MultiHeadMoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        moe_num_heads: int
        moe_head_hidden_size: int
        experts_per_head: int
        top_k: int
        moe_expert_inter_mult: int = 4
        moe_shared_inter_dim: int
        score_func: str = "sigmoid"
        route_norm: bool = True
        route_scale: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        if config.hidden_size % config.moe_num_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} must be divisible by "
                f"moe_num_heads {config.moe_num_heads}"
            )
        self.hidden_size = config.hidden_size
        self.moe_num_heads = config.moe_num_heads
        self.head_hidden_size = config.moe_head_hidden_size
        self.experts_per_head = config.experts_per_head
        self.top_k = config.top_k
        self.flatten_num_experts = config.moe_num_heads * config.experts_per_head
        self._head_groups: tuple = ()
        self._local_num_heads = config.moe_num_heads
        self._local_head_start = 0
        self._local_num_experts = self.flatten_num_experts
        self.gate = MultiHeadMoEGate(
            moe_num_heads=config.moe_num_heads,
            head_hidden_size=config.moe_head_hidden_size,
            experts_per_head=config.experts_per_head,
            top_k=config.top_k,
            score_func=config.score_func,
            route_norm=config.route_norm,
            route_scale=config.route_scale,
        )
        self.gate.local_head_start = 0
        self.gate.local_num_heads = config.moe_num_heads

        self.proj_in = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.moe_num_heads * config.moe_head_hidden_size,
            bias=False,
        ).build()
        self.proj_out = Linear.Config(
            in_features=config.moe_num_heads * config.moe_head_hidden_size,
            out_features=config.hidden_size,
            bias=False,
        ).build()
        self.gate = MultiHeadMoEGate(
            moe_num_heads=config.moe_num_heads,
            head_hidden_size=config.moe_head_hidden_size,
            experts_per_head=config.experts_per_head,
            top_k=config.top_k,
            score_func=config.score_func,
            route_norm=config.route_norm,
            route_scale=config.route_scale,
        )
        self.experts = MMGcGroupedExperts(
            num_experts=self.flatten_num_experts,
            hidden_size=config.moe_head_hidden_size,
            intermediate_size=config.moe_head_hidden_size * config.moe_expert_inter_mult,
        )
        self.experts.local_num_experts = self.flatten_num_experts
        self.shared_gate_proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.moe_shared_inter_dim,
            bias=False,
        ).build()
        self.shared_up_proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.moe_shared_inter_dim,
            bias=False,
        ).build()
        self.shared_down_proj = Linear.Config(
            in_features=config.moe_shared_inter_dim,
            out_features=config.hidden_size,
            bias=False,
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        w_in = _local_weight(self.proj_in.weight)
        x_heads = F.linear(x_flat, w_in).view(
            num_tokens, self._local_num_heads, self.head_hidden_size
        )
        topk_probs, topk_indices = self.gate(x_heads)

        head_offset = (
            torch.arange(self._local_num_heads, device=x_heads.device)
            .view(1, self._local_num_heads, 1)
            * self.experts_per_head
        )
        flat_expert_indices = (topk_indices + head_offset).reshape(-1)
        sorted_order = torch.argsort(flat_expert_indices, stable=True)
        gather_ids = sorted_order // self.top_k
        probs_sorted = topk_probs.reshape(-1)[sorted_order]
        num_tokens_per_expert = torch.zeros(
            self._local_num_experts,
            dtype=torch.int64,
            device=flat_expert_indices.device,
        ).scatter_add_(
            0,
            flat_expert_indices,
            torch.ones_like(flat_expert_indices),
        )

        routed_input = x_heads.reshape(
            num_tokens * self._local_num_heads, -1
        )[gather_ids]
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = routed_output * probs_sorted.unsqueeze(-1).to(
            routed_output.dtype
        )

        combined = torch.zeros(
            num_tokens * self._local_num_heads,
            self.head_hidden_size,
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        combined.index_add_(0, gather_ids, routed_output)

        w_out = _local_weight(self.proj_out.weight)
        moe_out = F.linear(
            combined.reshape(
                num_tokens, self._local_num_heads * self.head_hidden_size
            ),
            w_out,
        )
        if self._head_groups:
            for group in self._head_groups:
                moe_out = dist_nn.all_reduce(moe_out, group=group)

        shared_out = self.shared_down_proj(
            F.silu(self.shared_gate_proj(x_flat)) * self.shared_up_proj(x_flat)
        )
        return (moe_out + shared_out).view(original_shape)
