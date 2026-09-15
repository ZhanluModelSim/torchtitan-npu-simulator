# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only Block Diffusion fused attention used during meta simulation."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.block_diffusion_attention import BlockDiffusionAttentionModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

_original_converter: type | None = None
_TORCH_MAX_INT = 2_147_483_647


def _empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def _module_path() -> str:
    capture = get_active_capture()
    if capture is None or capture.module_path_tracker is None:
        return ""
    return capture.module_path_tracker.current_path()


def _record(
    raw_op_type: str,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    module_path: str,
    attrs: dict[str, int | float | str],
) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type,
            inputs=inputs,
            outputs=outputs,
            module_path=module_path,
            attrs=attrs,
        )


class _SimFusionAttention(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        sparse_mode,
        head_num,
        kv_head_num,
        head_dim,
        compute_alpha,
        module_path,
    ):
        output = _empty_like(q)
        attrs = {
            "num_heads": int(head_num),
            "head_num": int(head_num),
            "num_kv_heads": int(kv_head_num),
            "head_dim": int(head_dim),
            "layout": "BSH",
            "input_layout": "BSH",
            "scale": float(scale),
            "scale_value": float(scale),
            "keep_prob": 1.0,
            "pre_tokens": _TORCH_MAX_INT,
            "pre_tockens": _TORCH_MAX_INT,
            "next_tokens": 0,
            "next_tockens": 0,
            "inner_precise": 0,
            "sparse_mode": int(sparse_mode),
            "is_causal": True,
            "q_seq_len": int(q.shape[1]),
            "kv_seq_len": int(k.shape[1]),
            "gen_mask_parallel": True,
            "sync": False,
            "compute_alpha": float(compute_alpha),
        }
        _record("npu.npu_fusion_attention.default", [q, k, v], [output], module_path, attrs)
        ctx.save_for_backward(q, k, v)
        ctx.module_path = module_path
        ctx.attrs = attrs
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        grads = [_empty_like(tensor) for tensor in (q, k, v)]
        _record(
            "npu.npu_fusion_attention_grad.default",
            [q, k, v, grad_output],
            grads,
            ctx.module_path,
            ctx.attrs,
        )
        return *grads, None, None, None, None, None, None, None


class SimBlockDiffusionAttention(ScaledCausalSDPA):
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
        resolved_scale = float(scale) if scale is not None else q.shape[-1] ** -0.5
        output = _SimFusionAttention.apply(
            q.reshape(batch_size, query_len, -1),
            k.reshape(batch_size, key_len, -1),
            v.reshape(batch_size, key_len, -1),
            resolved_scale,
            2 if causal else 0,
            query_heads,
            k.shape[2],
            q.shape[-1],
            self.compute_alpha,
            _module_path(),
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
        if attention_masks is not None or kwargs:
            raise ValueError("simulated Block Diffusion attention only supports its internal visibility pattern")
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("SimBlockDiffusionAttention requires equal Q/K/V sequence lengths")
        if enable_gqa and q.shape[2] % k.shape[2] != 0:
            raise ValueError("query heads must be divisible by key/value heads for GQA")

        return self._fused_attention(q, k, v, scale=scale, causal=True)


class SimBlockDiffusionAttentionConverter(ModelCustomConverter):
    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if name and type(module) is ScaledCausalSDPA:
                replacement = SimBlockDiffusionAttention(
                    ScaledCausalSDPA.Config(compute_alpha=module.compute_alpha)
                )
                replace_module_with_name(model, name, replacement)


def apply_block_diffusion_attention_shim() -> None:
    global _original_converter
    if _original_converter is None:
        _original_converter = BlockDiffusionAttentionModelConfig.model_converter
    BlockDiffusionAttentionModelConfig.model_converter = SimBlockDiffusionAttentionConverter


def unapply_block_diffusion_attention_shim() -> None:
    global _original_converter
    if _original_converter is not None:
        BlockDiffusionAttentionModelConfig.model_converter = _original_converter
        _original_converter = None
