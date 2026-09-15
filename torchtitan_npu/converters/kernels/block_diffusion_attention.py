# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-sequence causal fused attention for Block Diffusion simulation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch_npu

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import ModelCustomConfig, ModelCustomConverter
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA

_COMPRESSED_CAUSAL_MASK_SIZE = 2048
_TORCH_MAX_INT = 2_147_483_647


def _compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Create the compressed upper-triangular mask required by sparse mode 2.

    The converter builds this once, outside the captured training step, and
    shares it across all Block Diffusion layers.  In the NPU API ``True``
    means hidden, which is the inverse of PyTorch SDPA's boolean convention.
    """

    return torch.triu(
        torch.ones(
            (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


class NPUBlockDiffusionAttention(ScaledCausalSDPA):
    """Execute one full-sequence causal NPU fused-attention kernel."""

    def __init__(self, parent: ScaledCausalSDPA, causal_mask: torch.Tensor) -> None:
        super().__init__(ScaledCausalSDPA.Config(compute_alpha=parent.compute_alpha))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    @staticmethod
    def _resolved_scale(q: torch.Tensor, scale: float | None) -> float:
        return float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])

    def _fused_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        batch_size, query_len, query_heads, _query_dim = q.shape
        key_len, value_dim = k.shape[1], v.shape[-1]
        # BSH is an officially supported equivalent NPU layout and is also
        # understood by downstream FA cost models that do not parse 4-D BSND.
        # These reshapes only change metadata; Sq and Skv remain independent.
        q_bsh = q.reshape(batch_size, query_len, -1)
        k_bsh = k.reshape(batch_size, key_len, -1)
        v_bsh = v.reshape(batch_size, key_len, -1)
        output = _NPUFusionAttention.apply(
            q_bsh,
            k_bsh,
            v_bsh,
            self.causal_mask if causal else None,
            self._resolved_scale(q, scale),
            2 if causal else 0,
            query_heads,
        )
        return output.reshape(batch_size, query_len, query_heads, value_dim)

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_masks=None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "NPUBlockDiffusionAttention constructs its visibility pattern internally; "
                "an external attention mask is not supported"
            )
        if kwargs:
            raise TypeError(f"unsupported fused-attention keyword arguments: {sorted(kwargs)}")
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("NPUBlockDiffusionAttention requires equal Q/K/V sequence lengths")
        if enable_gqa and q.shape[2] % k.shape[2] != 0:
            raise ValueError("query heads must be divisible by key/value heads for GQA")

        return self._fused_attention(q, k, v, scale=scale, causal=True)


class _NPUFusionAttention(torch.autograd.Function):
    """Autograd bridge for torch_npu's explicit fused-attention grad op."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, q, k, v, atten_mask, scale, sparse_mode, head_num):
        attention, softmax_max, softmax_sum, _softmax_out, seed, offset, numels = (
            torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=head_num,
                input_layout="BSH",
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=1.0,
                pre_tockens=_TORCH_MAX_INT,
                next_tockens=0,
                inner_precise=0,
                sparse_mode=sparse_mode,
                gen_mask_parallel=True,
                sync=False,
            )
        )
        ctx.save_for_backward(q, k, v, attention, softmax_max, softmax_sum)
        ctx.atten_mask = atten_mask
        ctx.head_num = head_num
        ctx.scale = scale
        ctx.sparse_mode = sparse_mode
        ctx.seed = seed
        ctx.offset = offset
        ctx.numels = numels
        return attention

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        q, k, v, attention, softmax_max, softmax_sum = ctx.saved_tensors
        grad_q, grad_k, grad_v, *_ = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            grad_output,
            head_num=ctx.head_num,
            input_layout="BSH",
            atten_mask=ctx.atten_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attention,
            scale_value=ctx.scale,
            keep_prob=1.0,
            pre_tockens=_TORCH_MAX_INT,
            next_tockens=0,
            inner_precise=0,
            seed=ctx.seed,
            offset=ctx.offset,
            numels=ctx.numels,
            sparse_mode=ctx.sparse_mode,
            gen_mask_parallel=True,
            sync=False,
        )
        return grad_q, grad_k, grad_v, None, None, None, None


class NPUBlockDiffusionAttentionConverter(ModelCustomConverter):
    def convert(self, model: nn.Module) -> None:
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if name and type(module) is ScaledCausalSDPA
        ]
        if not modules:
            return

        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        causal_mask = _compressed_causal_mask(device)
        for name, module in modules:
            replace_module_with_name(model, name, NPUBlockDiffusionAttention(module, causal_mask))


@register_model_converter("npu_block_diffusion_attention")
class BlockDiffusionAttentionModelConfig(ModelCustomConfig):
    model_converter = NPUBlockDiffusionAttentionConverter
