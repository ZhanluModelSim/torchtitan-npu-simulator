# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Next feed-forward modules: dense MLP (upstream) and sparse MoE.

The MoE block strictly mirrors the MindSpeed-LLM fsdp2 implementation
(``mindspeed_llm/fsdp2/models/qwen3_next/modeling_qwen3_next.py``):

- ``Qwen3NextMoeExperts``: fused ``gate_up_proj`` [E*hidden, 2*inter] /
  ``down_proj`` [E*inter, hidden] parameters and the grouped-GEMM ->
  ``npu_swiglu`` -> grouped-GEMM core. Token permute/unpermute live at the
  MoE boundary so the torchtitan ExpertParallel dispatcher (which wraps the
  experts module with the ``(routed_input, num_tokens_per_expert)`` contract)
  stays intact; routing probs are folded into the permuted tokens before the
  first GEMM (numerically equivalent to MindSpeed folding them into the
  unpermute).
- ``Qwen3NextMoeSharedExpert``: dense SwiGLU MLP with a sigmoid gate.

The eager helpers below provide the same semantics with plain aten ops so the
model also runs without the fused NPU kernels; the simulator binds virtual
fused operators (``qwen3_next_converter``) so the captured workload matches
the MindSpeed production op stream.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


def _token_permute(
    hidden_states: torch.Tensor, selected_experts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager equivalent of ``npu_moe_token_permute`` (expert-major sort)."""
    flat_indices = selected_experts.reshape(-1).to(torch.int64)
    row_ids_map = torch.argsort(flat_indices, stable=True)
    permuted_hidden_states = hidden_states.index_select(0, row_ids_map)
    return permuted_hidden_states, row_ids_map


def _swiglu(fc1_output: torch.Tensor) -> torch.Tensor:
    """Eager equivalent of ``torch_npu.npu_swiglu(x, dim=-1)``."""
    gate, up = fc1_output.chunk(2, dim=-1)
    return F.silu(gate) * up


def _token_unpermute(
    routed_output: torch.Tensor,
    row_ids_map: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Eager equivalent of ``npu_moe_token_unpermute`` (probs pre-folded)."""
    token_ids = row_ids_map // top_k
    output = torch.zeros(
        (num_tokens, routed_output.shape[-1]),
        dtype=routed_output.dtype,
        device=routed_output.device,
    )
    output.index_add_(0, token_ids, routed_output)
    return output


class Qwen3NextGroupedExperts(nn.Module):
    """Routed experts mirroring MindSpeed ``Qwen3NextMoeExperts``.

    Fused weight layout: ``gate_up_proj`` [E*hidden, 2*inter] (viewed
    [E, hidden, 2*inter], trans_b=False in the GMM) and ``down_proj``
    [E*inter, hidden] (viewed [E, inter, hidden]). The expert dimension is
    dim 0 so ``ExpertParallel`` shards both parameters with Shard(0). The
    module keeps torchtitan's ``(routed_input, num_tokens_per_expert)``
    contract: token permute/unpermute are handled at the MoE boundary.
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts * hidden_size, 2 * intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts * intermediate_size, hidden_size)
        )

    def _view_experts_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if isinstance(self.gate_up_proj, DTensor)
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if isinstance(self.down_proj, DTensor)
            else self.down_proj
        )
        # The fused flat layout [E*hidden, 2*inter] shards row-wise under EP
        # (Shard(0)), which evenly splits whole experts; derive the local
        # expert count from the actual storage instead of self.num_experts.
        local_experts = gate_up_proj.shape[0] // self.hidden_size
        gate_up_proj = gate_up_proj.view(local_experts, self.hidden_size, -1)
        down_proj = down_proj.view(local_experts, -1, self.hidden_size)
        return gate_up_proj, down_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        gate_up_proj, down_proj = self._view_experts_weight()

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        fc1_output = torch._grouped_mm(hidden_states, gate_up_proj, offs=offsets)
        fc1_activation = _swiglu(fc1_output)
        fc2_out = torch._grouped_mm(fc1_activation, down_proj, offs=offsets)
        return fc2_out


class Qwen3NextMoE(nn.Module):
    """Sparse MoE block mirroring MindSpeed ``Qwen3NextSparseFusedMoeBlock``."""

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
        self.norm_topk_prob = config.norm_topk_prob
        self.debug_force_load_balance = config.debug_force_load_balance

        # gating
        self.gate = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        ).build()
        self.experts = Qwen3NextGroupedExperts(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )

        # shared expert
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

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # gating
        gate_weight = (
            self.gate.weight.to_local()
            if isinstance(self.gate.weight, DTensor)
            else self.gate.weight
        )
        router_logits = F.linear(hidden_flat, gate_weight)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        if self.debug_force_load_balance:
            token_offsets = torch.arange(
                hidden_flat.shape[0], device=hidden_flat.device, dtype=torch.int64
            ) * self.top_k
            expert_offsets = torch.arange(
                self.top_k, device=hidden_flat.device, dtype=torch.int64
            )
            selected_experts = (
                token_offsets.unsqueeze(1) + expert_offsets.unsqueeze(0)
            ) % self.num_experts
            routing_weights = torch.full(
                (hidden_flat.shape[0], self.top_k),
                1.0 / self.top_k,
                dtype=torch.float32,
                device=hidden_flat.device,
            )
        else:
            routing_weights, selected_experts = torch.topk(
                routing_weights, k=self.top_k, dim=-1
            )
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(
                    dim=-1, keepdim=True
                )
        routing_weights = routing_weights.to(hidden_states.dtype)

        # permute (npu_moe_token_permute semantics, probs pre-folded)
        permuted_hidden_states, row_ids_map = _token_permute(
            hidden_flat, selected_experts
        )
        tokens_per_expert = torch.histc(
            selected_experts.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        row_probs = routing_weights.reshape(-1).index_select(0, row_ids_map)
        permuted_hidden_states = (
            permuted_hidden_states * row_probs.to(permuted_hidden_states.dtype).unsqueeze(-1)
        )

        routed_output = self.experts(permuted_hidden_states, tokens_per_expert)

        # unpermute (npu_moe_token_unpermute semantics)
        combined = _token_unpermute(
            routed_output,
            row_ids_map,
            num_tokens=hidden_flat.shape[0],
            top_k=self.top_k,
        )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        combined = combined + shared_gate * shared_expert_output

        return combined.view(batch_size, sequence_length, hidden_dim)


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
