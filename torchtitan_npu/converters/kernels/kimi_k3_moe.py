# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU ACLNN MoE converter for Kimi K3's KimiSparseMoeBlock.

Replaces the standard per-expert loop with NPU fused operators:
- 3D tensor weights: gate_up_proj[E, H, 2*I], down_proj[E, I, H]
- torch._grouped_mm for batched expert matmul
- torch_npu.npu_swiglu for fused activation (adapted for SiTU-GLU)
- npu_moe_token_permute / npu_moe_token_unpermute for token dispatch

Reference: MindSpeed-MM kimi_moe_patch.py (PatchKimiMoeExperts)
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.kimi_k3.feed_forward import KimiSparseMoeBlock, SituGLU

logger = logging.getLogger(__name__)


class NpuKimiK3MoeExperts(nn.Module):
    """Routed experts as 3D tensors with NPU grouped GEMM.

    Weight layout (same as MindSpeed-MM PatchKimiMoeExperts):
        gate_up_proj: [num_experts, hidden_size, 2 * intermediate_size]
        down_proj:    [num_experts, intermediate_size, hidden_size]
    """

    def __init__(self, parent: KimiSparseMoeBlock):
        super().__init__()
        config = parent.config
        self.num_experts = config.num_experts
        self.hidden_size = parent.moe_hidden_size
        self.intermediate_size = config.moe_intermediate_size

        # Fuse per-expert weights into 3D tensors
        gate_up_data = torch.empty(
            self.num_experts, self.hidden_size, 2 * self.intermediate_size
        )
        down_data = torch.empty(
            self.num_experts, self.intermediate_size, self.hidden_size
        )

        for i, expert in enumerate(parent.experts):
            gate_up_data[i] = torch.cat([expert.w1.weight, expert.w3.weight], dim=0).T
            down_data[i] = expert.w2.weight.T

        self.gate_up_proj = nn.Parameter(gate_up_data)
        self.down_proj = nn.Parameter(down_data)

        # Activation
        self.act_fn = SituGLU(beta=config.situ_beta, linear_beta=config.situ_linear_beta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weight: torch.Tensor,
    ) -> torch.Tensor:
        """NPU fused MoE forward using grouped_mm + permute/unpermute."""
        import torch_npu

        num_tokens = hidden_states.shape[0]

        # Token permute: sort tokens by expert assignment
        sorted_indices = top_k_index.view(-1).argsort()
        token_indices = sorted_indices // top_k_index.shape[1]
        permuted_hidden = hidden_states[token_indices]

        # Compute tokens per expert
        flat_experts = top_k_index.view(-1)
        num_tokens_per_expert = torch.histc(
            flat_experts.float(), bins=self.num_experts, min=0, max=self.num_experts - 1
        ).to(torch.int64)

        # Grouped GEMM: gate_up
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)
        gate_up = torch._grouped_mm(
            permuted_hidden.bfloat16(),
            self.gate_up_proj.bfloat16().transpose(-2, -1),
            offs=offsets,
        )

        # Activation (SiTU-GLU)
        activated = self.act_fn(gate_up)

        # Grouped GEMM: down
        output = torch._grouped_mm(
            activated.bfloat16(),
            self.down_proj.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(hidden_states)

        # Token unpermute: scatter back with routing weights
        routing_weights = top_k_weight.view(-1)[sorted_indices.argsort()]
        output = output * routing_weights.unsqueeze(-1)

        # Aggregate
        final = torch.zeros(num_tokens, self.hidden_size, dtype=output.dtype, device=output.device)
        final.index_add_(0, token_indices, output)

        return final


class NpuKimiK3MoeBlock(KimiSparseMoeBlock):
    """KimiSparseMoeBlock with NPU fused expert computation.

    Replaces the per-expert ModuleList with 3D tensor grouped GEMM.
    LatentMoE projections and shared experts remain unchanged.
    """

    def __init__(self, parent: KimiSparseMoeBlock):
        # Copy all attributes from parent
        self.__dict__.update(parent.__dict__)
        # Replace experts with fused 3D tensor version
        self.npu_experts = NpuKimiK3MoeExperts(parent)
        # Remove original per-expert list to save memory
        del self.experts

    def _moe_forward(
        self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        return self.npu_experts(x, topk_idx, topk_weight)


class NpuKimiK3MoeConverter(ModelCustomConverter):
    """Replaces KimiSparseMoeBlock with NpuKimiK3MoeBlock (3D tensor + grouped_mm)."""

    def convert(self, model: nn.Module) -> None:
        replaced = 0
        for name, module in model.named_modules():
            if isinstance(module, KimiSparseMoeBlock) and not isinstance(module, NpuKimiK3MoeBlock):
                replace_module_with_name(model, name, NpuKimiK3MoeBlock(module))
                replaced += 1
        if replaced:
            logger.info(f"[npu_kimi_k3_moe] Replaced {replaced} KimiSparseMoeBlock with NPU fused experts")


@register_model_converter("npu_kimi_k3_moe")
class NpuKimiK3MoeModelConfig:
    model_converter = NpuKimiK3MoeConverter
