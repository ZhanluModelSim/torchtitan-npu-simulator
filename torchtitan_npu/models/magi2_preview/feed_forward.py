# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview feed-forward modules: swiglu7, dense MLP, multi-head MoE.

Ports the feed-forward stack of the official MAGI-2-preview inference model
(``MLP``, ``CoreMultiHeadMoE``, ``MultiHeadMoELayer``) to plain, autograd
friendly torch for training. Attribute names match the official checkpoint
keys exactly (see the ``block.layers.{i}.mlp.*`` tree in the port design
contract); all parameter/buffer values are filled later by the model-level
``init_weights`` (or by loading the official checkpoint).
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module

from .grouped_linear import GroupedLinear
from .norms import MultiModalityRMSNorm

logger = logging.getLogger(__name__)


def swiglu7(
    x: torch.Tensor,
    up: torch.Tensor | None = None,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Clamped SwiGLU activation (GPT-OSS style, +1 bias on the linear part).

    Two input forms are supported:

    * Interleaved (``up is None``): the last dimension of ``x`` carries
      interleaved gate/up channels, ``g = x[..., 0::2]`` and
      ``u = x[..., 1::2]`` (used by the dense MLP and shared experts).
    * Pair (``up`` given): ``x`` is the gate tensor and ``up`` the linear
      tensor (used by the routed MoE experts).

    Math (computed in fp32): ``g = clamp(g, max=limit)`` (one-sided),
    ``u = clamp(u, -limit, limit)``, then
    ``out = (g * sigmoid(alpha * g)) * (u + 1)``.
    """
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x_f = x.to(torch.float32)
    if up is None:
        gate = x_f[..., 0::2]
        linear = x_f[..., 1::2]
    else:
        gate = x_f
        linear = up.to(torch.float32)
    gate = gate.clamp(max=limit)
    linear = linear.clamp(min=-limit, max=limit)
    out = (gate * torch.sigmoid(alpha * gate)) * (linear + 1)
    return out.to(out_dtype)


class Magi2MLP(Module):
    """Dense MLP used on the modality-mixing (mm) layers.

    ``pre_norm -> up_gate_proj -> swiglu7 -> down_proj`` with per-modality
    grouped linears. Checkpoint keys: ``mlp.pre_norm.weight``,
    ``mlp.up_gate_proj.weight``, ``mlp.down_proj.weight``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 3072
        intermediate_size: int = 8192
        num_modality: int = 3
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size,
            num_modality=config.num_modality,
            eps=config.norm_eps,
        )
        self.up_gate_proj = GroupedLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            config.num_modality,
        )
        self.down_proj = GroupedLinear(
            config.intermediate_size,
            config.hidden_size,
            config.num_modality,
        )

    def forward(
        self, x: torch.Tensor, m_splits: list[int] | None = None
    ) -> torch.Tensor:
        x = self.pre_norm(x, m_splits)
        x = self.up_gate_proj(x, m_splits)
        x = swiglu7(x)
        x = self.down_proj(x, m_splits)
        return x


class Magi2Router(nn.Module):
    """Sigmoid top-k router for multi-head MoE with aux-free expert bias.

    Holds the ``expert_bias`` / ``expert_bias_ema`` buffers required by the
    official checkpoint layout (``moe_mlp.router.*``). Only ``expert_bias``
    takes part in routing, and only for expert *selection*: the returned
    probabilities are gathered from the unbiased sigmoid scores.
    """

    def __init__(
        self,
        flatten_num_experts: int,
        top_k: int,
        route_norm: bool = True,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.register_buffer(
            "expert_bias",
            torch.zeros(flatten_num_experts, dtype=torch.float32),
        )
        self.register_buffer(
            "expert_bias_ema",
            torch.zeros(flatten_num_experts, dtype=torch.float32),
        )

    def forward(
        self, x_heads: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route per-head token embeddings.

        Args:
            x_heads: token embeddings of shape ``(T, H, d_head)``.
            gate: router weight of shape ``(H * E, d_head)``.

        Returns:
            ``(topk_probs, topk_indices)`` with shapes ``(H, T, K)``;
            probs are fp32, scaled by ``route_scale``.
        """
        num_heads = x_heads.shape[1]
        num_experts = gate.shape[0] // num_heads
        gate_f = gate.view(num_heads, num_experts, -1).float()
        logits = torch.einsum("shd,hed->hse", x_heads.float(), gate_f)
        scores = torch.sigmoid(logits)
        biased = scores + self.expert_bias.view(num_heads, 1, num_experts)
        topk_indices = biased.topk(self.top_k, dim=-1).indices
        # Unbiased probs: gather from the scores *without* expert_bias.
        topk_probs = scores.gather(-1, topk_indices)
        if self.route_norm:
            topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=1e-12)
        topk_probs = topk_probs * self.route_scale
        return topk_probs, topk_indices


class CoreMultiHeadMoE(Module):
    """Multi-head MoE core: per-head sigmoid routing over flattened experts.

    Expert weights are stored flattened, expert-major: ``gate`` is
    ``(H * E, d_head)`` fp32, ``W_gate`` / ``W_up`` are
    ``(H * E, d_head, d_expert)`` and ``W_down`` is
    ``(H * E, d_expert, d_head)``; global expert id is ``h * E + e``.
    Tokens are grouped per head by expert id (argsort + segment boundaries)
    and accumulated with ``index_add_`` so the whole path is autograd
    friendly. Routing runs in fp32; expert matmuls run in parameter dtype.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 3072
        num_heads: int = 12
        num_experts: int = 256
        top_k: int = 6
        expert_intermediate_size: int = 1280
        route_norm: bool = True
        route_scale: float = 4.9

    def __init__(self, config: Config):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.d_head = config.hidden_size // config.num_heads
        self.d_expert = config.expert_intermediate_size
        flatten_num_experts = self.num_heads * self.num_experts

        self.gate = nn.Parameter(
            torch.empty(flatten_num_experts, self.d_head, dtype=torch.float32)
        )
        self.W_gate = nn.Parameter(
            torch.empty(flatten_num_experts, self.d_head, self.d_expert)
        )
        self.W_up = nn.Parameter(
            torch.empty(flatten_num_experts, self.d_head, self.d_expert)
        )
        self.W_down = nn.Parameter(
            torch.empty(flatten_num_experts, self.d_expert, self.d_head)
        )
        self.router = Magi2Router(
            flatten_num_experts,
            top_k=self.top_k,
            route_norm=self.route_norm,
            route_scale=self.route_scale,
        )

    def _expert_forward(
        self,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Expert matmuls run in parameter dtype (routing stays fp32).
        x_heads = x_heads.to(self.W_gate.dtype)
        num_tokens = x_heads.shape[0]
        out = torch.zeros(
            (num_tokens, self.num_heads, self.d_head),
            dtype=torch.float32,
            device=x_heads.device,
        )
        token_ids = torch.arange(
            num_tokens, device=x_heads.device
        ).repeat_interleave(self.top_k)
        for h in range(self.num_heads):
            expert_ids = topk_indices[h].reshape(-1)
            probs = topk_probs[h].reshape(-1)
            order = expert_ids.argsort(stable=True)
            sorted_experts = expert_ids[order]
            sorted_probs = probs[order]
            sorted_tokens = token_ids[order]
            counts = torch.bincount(sorted_experts, minlength=self.num_experts)
            ends = counts.cumsum(0).tolist()
            start = 0
            for e, end in enumerate(ends):
                if end == start:
                    continue
                toks = sorted_tokens[start:end]
                p = sorted_probs[start:end]
                expert_id = h * self.num_experts + e
                xs = x_heads[toks, h, :]
                h_act = swiglu7(
                    xs @ self.W_gate[expert_id], xs @ self.W_up[expert_id]
                )
                contrib = h_act.to(xs.dtype) @ self.W_down[expert_id]
                out[:, h, :].index_add_(
                    0, toks, contrib.float() * p.unsqueeze(-1)
                )
                start = end
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_heads = x.view(-1, self.num_heads, self.d_head)
        topk_probs, topk_indices = self.router(x_heads, self.gate)
        out = self._expert_forward(x_heads, topk_probs, topk_indices)
        return out.to(x.dtype).reshape(-1, self.num_heads * self.d_head)


class MultiHeadMoELayer(Module):
    """MoE block of the middle layers: routed core plus shared experts.

    ``pre_norm`` output feeds the routed path (``split_linear -> moe_mlp ->
    merge_linear``) and the shared path in parallel. The shared path fuses
    the modality-agnostic and modality-specific experts: both fc1 outputs
    are concatenated, passed through one interleaved swiglu7, then split
    back for the two fc2 projections. Checkpoint keys: ``mlp.pre_norm``,
    ``mlp.split_linear``, ``mlp.merge_linear``, ``mlp.moe_mlp.*``,
    ``mlp.shared_expert_fc1/fc2``, ``mlp.modality_specific_shared_expert_fc1/fc2``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 3072
        num_modality: int = 3
        moe_num_heads: int = 12
        num_experts: int = 256
        moe_top_k: int = 6
        expert_intermediate_size: int = 1280
        shared_expert_intermediate_size: int = 1280
        route_norm: bool = True
        route_scale: float = 4.9
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        inter = config.shared_expert_intermediate_size
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size,
            num_modality=config.num_modality,
            eps=config.norm_eps,
        )
        self.split_linear = GroupedLinear(config.hidden_size, config.hidden_size, 1)
        self.merge_linear = GroupedLinear(config.hidden_size, config.hidden_size, 1)
        self.moe_mlp = CoreMultiHeadMoE(
            CoreMultiHeadMoE.Config(
                hidden_size=config.hidden_size,
                num_heads=config.moe_num_heads,
                num_experts=config.num_experts,
                top_k=config.moe_top_k,
                expert_intermediate_size=config.expert_intermediate_size,
                route_norm=config.route_norm,
                route_scale=config.route_scale,
            )
        )
        self.shared_expert_fc1 = GroupedLinear(config.hidden_size, inter * 2, 1)
        self.shared_expert_fc2 = GroupedLinear(inter, config.hidden_size, 1)
        self.modality_specific_shared_expert_fc1 = GroupedLinear(
            config.hidden_size, inter * 2, config.num_modality
        )
        self.modality_specific_shared_expert_fc2 = GroupedLinear(
            inter, config.hidden_size, config.num_modality
        )
        self._shared_intermediate_size = inter

    def _shared_expert_forward(
        self, norm_output: torch.Tensor, m_splits: list[int] | None
    ) -> torch.Tensor:
        x1 = self.shared_expert_fc1(norm_output, None)
        x2 = self.modality_specific_shared_expert_fc1(norm_output, m_splits)
        x = swiglu7(torch.cat([x1, x2], dim=-1))
        x1, x2 = x.split([self._shared_intermediate_size] * 2, dim=-1)
        x1 = self.shared_expert_fc2(x1, None)
        x2 = self.modality_specific_shared_expert_fc2(x2.contiguous(), m_splits)
        return x1 + x2

    def forward(
        self, x: torch.Tensor, m_splits: list[int] | None = None
    ) -> torch.Tensor:
        norm_output = self.pre_norm(x, m_splits)
        moe_out = self.merge_linear(
            self.moe_mlp(self.split_linear(norm_output, None)), None
        )
        return moe_out + self._shared_expert_forward(norm_output, m_splits)
