# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import types
from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan_npu.models.deepseek_v4.model import Compressor, apply_partial_rotary_emb_

from ..model_custom_interface import ModelCustomConfig, ModelCustomConverter
from ..registry import register_model_converter


class CompressorFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        module,
        x,
        flat_positions,
        block_starts,
        cu_seqlens_q,
        seqused,
        start_pos,
        state_block_table,
        state_cache,
        wkv,
        wgate,
        ape,
    ):
        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        coff = 1 + int(module.overlap)
        output, softmax_score, kv = module._compressor_forward_op(
            flat_x,
            wkv,
            wgate,
            state_cache,
            ape,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens_q,
            seqused=seqused,
            start_pos=start_pos,
            cmp_ratio=module.compress_ratio,
            coff=coff,
            cache_mode=1,
            grad_enabled=True,
        )
        ctx.input_shape = x.shape
        ctx.compress_ratio = module.compress_ratio
        ctx.head_dim = module.head_dim
        ctx.overlap = module.overlap
        ctx.x_dtype = x.dtype
        ctx.wkv_dtype = wkv.dtype
        ctx.wgate_dtype = wgate.dtype
        ctx.ape_dtype = ape.dtype
        ctx.save_for_backward(flat_x, wkv, wgate, softmax_score, kv, flat_positions, block_starts)
        return output[: block_starts.numel()]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        fused_x, fused_wkv, fused_wgate, softmax_score, kv, flat_positions, block_starts = ctx.saved_tensors
        ratio = ctx.compress_ratio
        head_dim = ctx.head_dim
        block_indices = block_starts.unsqueeze(1) + torch.arange(ratio, device=block_starts.device).unsqueeze(0)
        num_blocks = block_starts.numel()
        grouped_softmax = softmax_score[:num_blocks].float()
        grouped_kv = kv[:num_blocks].float()
        grad_output = grad_output.float()

        grad_value = grouped_softmax * grad_output.unsqueeze(1)
        grad_probability = grouped_kv * grad_output.unsqueeze(1)
        grad_logits = grouped_softmax * (
            grad_probability - (grad_probability * grouped_softmax).sum(dim=1, keepdim=True)
        )

        flat_block_indices = block_indices.flatten()
        if ctx.overlap:
            grad_value = grad_value.unflatten(1, (ratio, 2))
            grad_logits = grad_logits.unflatten(1, (ratio, 2))
            previous_grad_value, current_grad_value = grad_value.unbind(dim=2)
            previous_grad_logits, current_grad_logits = grad_logits.unbind(dim=2)
            previous_indices = (block_indices - ratio).clamp_min(0)
            block_positions = flat_positions[block_starts]
            previous_starts = (block_starts - ratio).clamp_min(0)
            has_previous = block_starts.ge(ratio) & flat_positions[previous_starts].eq(block_positions - ratio)
            valid_previous_indices = previous_indices[has_previous].flatten()
            # Encode the overlap branch in the scatter row so both projection gradients
            # accumulate directly into their final [tokens, 2 * head_dim] storage.
            previous_scatter_indices = valid_previous_indices * 2
            current_scatter_indices = flat_block_indices * 2 + 1
            projection_grads = fused_x.new_zeros(
                (2, fused_x.shape[0], 2, head_dim),
                dtype=torch.float32,
            )
            grad_kv, grad_score = projection_grads.unbind(dim=0)
            flat_grad_kv = grad_kv.flatten(0, 1)
            flat_grad_score = grad_score.flatten(0, 1)
            flat_grad_kv.index_add_(
                0,
                previous_scatter_indices,
                previous_grad_value[has_previous].flatten(0, 1),
            )
            flat_grad_score.index_add_(
                0,
                previous_scatter_indices,
                previous_grad_logits[has_previous].flatten(0, 1),
            )
            flat_grad_kv.index_add_(0, current_scatter_indices, current_grad_value.flatten(0, 1))
            flat_grad_score.index_add_(0, current_scatter_indices, current_grad_logits.flatten(0, 1))
            grad_kv = grad_kv.flatten(1)
            grad_score = grad_score.flatten(1)
        else:
            grad_kv = fused_x.new_zeros((fused_x.shape[0], head_dim), dtype=torch.float32)
            grad_score = torch.zeros_like(grad_kv)
            grad_kv.index_add_(0, flat_block_indices, grad_value.flatten(0, 1))
            grad_score.index_add_(0, flat_block_indices, grad_logits.flatten(0, 1))

        grad_ape = grad_logits.flatten(2).sum(dim=0) if ctx.overlap else grad_logits.sum(dim=0)

        input_fp32 = fused_x.float()
        grad_x = grad_kv @ fused_wkv.float() + grad_score @ fused_wgate.float()
        grad_wkv = grad_kv.t() @ input_fp32
        grad_wgate = grad_score.t() @ input_fp32
        return (
            None,
            grad_x.reshape(ctx.input_shape).to(ctx.x_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_wkv.to(ctx.wkv_dtype),
            grad_wgate.to(ctx.wgate_dtype),
            grad_ape.to(ctx.ape_dtype),
        )


def _forward_tnd_fused(self, x, freqs_cis, positions, attention_masks=None):
    attention_masks = cast("Any", attention_masks)
    ratio = self.compress_ratio
    block_starts = attention_masks.block_starts_by_ratio.get(ratio)
    compressor_metadata = attention_masks.compressor_metadata
    state_block_table = compressor_metadata.state_block_table[ratio]
    state_cache = compressor_metadata.state_cache[(ratio, self.head_dim)]
    if state_cache.dtype != torch.float32:
        state_cache = state_cache.to(dtype=torch.float32)
        compressor_metadata.state_cache[(ratio, self.head_dim)] = state_cache
    output = CompressorFunction.apply(
        self,
        x,
        compressor_metadata.flat_positions,
        block_starts,
        compressor_metadata.cu_seqlens_q,
        compressor_metadata.seqused,
        compressor_metadata.start_pos,
        state_block_table,
        state_cache,
        self.wkv.weight,
        self.wgate.weight,
        self.ape,
    )
    output = self.norm(output.to(x.dtype))
    return apply_partial_rotary_emb_(
        output,
        freqs_cis,
        partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        positions=compressor_metadata.compressed_positions[ratio],
    )


class NpuCompressorConverter(ModelCustomConverter):
    """Enable the CANN compressor forward with an explicit small-op backward."""

    def convert(self, model: nn.Module) -> None:
        if self.model_name != "deepseek_v4":
            return
        importlib.import_module("cann_ops_transformer.ops.compressor")
        compressor_op = torch.ops.cann_ops_transformer._compressor_forward.default
        for module in model.modules():
            if isinstance(module, Compressor):
                module.use_tnd_metadata = True
                cast("Any", module)._compressor_forward_op = compressor_op
                module._forward_tnd = types.MethodType(_forward_tnd_fused, module)


@register_model_converter("npu_compressor")
class CompressorModelConfig(ModelCustomConfig):
    model_converter = NpuCompressorConverter
