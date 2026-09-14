# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused NPU attention for Block Diffusion's prefix/canvas visibility."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch_npu

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import ModelCustomConfig, ModelCustomConverter
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.block_diffusion.attention import PrefixCanvasSDPA

_COMPRESSED_CAUSAL_MASK_SIZE = 2048


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


class NPUBlockDiffusionAttention(PrefixCanvasSDPA):
    """Execute the prefix and canvas as two NPU fused-attention kernels."""

    def __init__(self, parent: PrefixCanvasSDPA, causal_mask: torch.Tensor) -> None:
        super().__init__(PrefixCanvasSDPA.Config(block_size=parent.block_size))
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
        return _NPUFusionAttention.apply(
            q,
            k,
            v,
            self.causal_mask if causal else None,
            self._resolved_scale(q, scale),
            2 if causal else 0,
        )

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

        seq_len = q.shape[1]
        if seq_len < self.block_size or seq_len % self.block_size != 0:
            raise ValueError(
                "sequence length must be a multiple of block_size and contain "
                f"one canvas, got seq_len={seq_len}, block_size={self.block_size}"
            )

        prefix_len = seq_len - self.block_size
        outputs = []
        if prefix_len:
            outputs.append(
                self._fused_attention(
                    q[:, :prefix_len],
                    k[:, :prefix_len],
                    v[:, :prefix_len],
                    scale=scale,
                    causal=True,
                )
            )
        outputs.append(
            self._fused_attention(
                q[:, prefix_len:],
                k,
                v,
                scale=scale,
                causal=False,
            )
        )
        return torch.cat(outputs, dim=1) if prefix_len else outputs[0]


class _NPUFusionAttention(torch.autograd.Function):
    """Autograd bridge for torch_npu's explicit fused-attention grad op."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, q, k, v, atten_mask, scale, sparse_mode):
        attention, softmax_max, softmax_sum, _softmax_out, seed, offset, numels = (
            torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[2],
                input_layout="BSND",
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=1.0,
                sparse_mode=sparse_mode,
            )
        )
        ctx.save_for_backward(q, k, v, attention, softmax_max, softmax_sum)
        ctx.atten_mask = atten_mask
        ctx.head_num = q.shape[2]
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
            input_layout="BSND",
            atten_mask=ctx.atten_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attention,
            scale_value=ctx.scale,
            keep_prob=1.0,
            seed=ctx.seed,
            offset=ctx.offset,
            numels=ctx.numels,
            sparse_mode=ctx.sparse_mode,
        )
        return grad_q, grad_k, grad_v, None, None, None


class NPUBlockDiffusionAttentionConverter(ModelCustomConverter):
    def convert(self, model: nn.Module) -> None:
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if name and type(module) is PrefixCanvasSDPA
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
