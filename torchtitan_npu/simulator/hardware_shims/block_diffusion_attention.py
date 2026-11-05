# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only Block Diffusion attention used during meta simulation.

The downstream analytical model cannot currently cost the synthetic
``npu_fusion_attention`` node.  Record the fused kernel as the canonical
matmul/softmax decomposition instead, while keeping the original fused
metadata on every child node.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.block_diffusion_attention import BlockDiffusionAttentionModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.block_diffusion.attention import ScaledCausalSDPA, attention_segments
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

_original_converter: type | None = None
_TORCH_MAX_INT = 2_147_483_647


def _empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def _empty(shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    """Allocate a shape-only placeholder without leaking an ``empty`` event."""
    capture = get_active_capture()
    if capture is None:
        return torch.empty(shape, dtype=reference.dtype, device=reference.device)
    with capture.suspend_recording():
        return torch.empty(shape, dtype=reference.dtype, device=reference.device)


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
    attrs: dict[str, int | float | str | bool],
    parameter_inputs: dict[str, int | float | str | bool] | None = None,
    dependency_inputs: list[torch.Tensor] | None = None,
) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type,
            inputs=inputs,
            outputs=outputs,
            module_path=module_path,
            attrs=attrs,
            parameter_inputs=parameter_inputs,
            dependency_inputs=dependency_inputs,
        )


def _heads_to_bsh(
    heads: torch.Tensor,
    module_path: str,
    metadata: dict[str, int | float | str | bool],
) -> torch.Tensor:
    """Record real BHSD -> BSH layout transitions, retaining actual aliases."""
    batch, num_heads, sequence, dim = heads.shape
    capture = get_active_capture()
    with capture.suspend_recording() if capture is not None else nullcontext():
        transposed = heads.transpose(1, 2)
        physical = transposed.reshape(batch, sequence, num_heads * dim)
    transpose_metadata = {**metadata, "dim0": 1, "dim1": 2}
    _record("aten.transpose.int", [heads], [transposed], module_path, transpose_metadata, transpose_metadata)
    _record("aten.reshape.default", [transposed], [physical], module_path, metadata, metadata)
    return physical


def _per_head_to_bsh(
    per_head: torch.Tensor,
    batch: int,
    num_heads: int,
    module_path: str,
    metadata: dict[str, int | float | str | bool],
) -> torch.Tensor:
    sequence, dim = per_head.shape[-2:]
    capture = get_active_capture()
    with capture.suspend_recording() if capture is not None else nullcontext():
        heads = per_head.reshape(batch, num_heads, sequence, dim)
    _record("aten.view.default", [per_head], [heads], module_path, metadata, metadata)
    return _heads_to_bsh(heads, module_path, metadata)


def _kv_gradient_to_bsh(
    per_head: torch.Tensor,
    batch: int,
    query_heads: int,
    kv_heads: int,
    module_path: str,
    metadata: dict[str, int | float | str | bool],
) -> torch.Tensor:
    """Reduce GQA contributions before converting physical KV gradients."""
    if query_heads == kv_heads:
        return _per_head_to_bsh(per_head, batch, kv_heads, module_path, metadata)
    sequence, dim = per_head.shape[-2:]
    capture = get_active_capture()
    with capture.suspend_recording() if capture is not None else nullcontext():
        grouped = per_head.reshape(batch, kv_heads, query_heads // kv_heads, sequence, dim)
        reduced = grouped.sum(dim=2)
    _record("aten.view.default", [per_head], [grouped], module_path, metadata, metadata)
    reduction_metadata = {**metadata, "reduction_dim": 2, "reduction": "gqa_query_head_groups"}
    _record("aten.sum.dim_IntList", [grouped], [reduced], module_path, reduction_metadata, reduction_metadata)
    return _heads_to_bsh(reduced, module_path, metadata)


def _fused_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    head_dim: int,
    scale: float,
    sparse_mode: int,
    compute_alpha: float,
) -> dict[str, int | float | str | bool]:
    """Describe the fused FA boundary retained on each decomposed sub-op."""
    return {
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
        "next_tokens": 0 if sparse_mode == 2 else _TORCH_MAX_INT,
        "next_tockens": 0 if sparse_mode == 2 else _TORCH_MAX_INT,
        "inner_precise": 0,
        "sparse_mode": int(sparse_mode),
        "is_causal": sparse_mode == 2,
        "q_seq_len": int(q.shape[1]),
        "kv_seq_len": int(k.shape[1]),
        "gen_mask_parallel": True,
        "sync": False,
        "compute_alpha": float(compute_alpha),
        "decomposed_from": "npu.npu_fusion_attention.default",
    }


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
        attention_mode,
        module_path,
    ):
        batch = q.shape[0]
        q_seq, k_seq = q.shape[1], k.shape[1]
        value_head_dim = v.shape[-1] // kv_head_num
        bh = batch * head_num

        # Approximate causal AR work with half of the full key sequence.
        # Fold Block Diffusion's alpha into that effective key dimension so
        # downstream matmul/softmax models apply it even if they ignore attrs.
        if attention_mode == "exact":
            effective_key_seq = k_seq
        else:
            causal_key_seq = (k_seq + 1) // 2 if sparse_mode == 2 else k_seq
            effective_key_seq = max(1, round(causal_key_seq * compute_alpha))

        q_per_head = _empty((bh, q_seq, head_dim), q)
        k_transposed = _empty((bh, head_dim, effective_key_seq), k)
        scores = _empty((bh, q_seq, effective_key_seq), q)
        probabilities = _empty_like(scores)
        v_per_head = _empty((bh, effective_key_seq, value_head_dim), v)
        output = _empty((bh, q_seq, value_head_dim), q) if attention_mode == "exact" else _empty_like(q)
        metadata = _fused_metadata(q, k, head_num, kv_head_num, head_dim, scale, sparse_mode, compute_alpha)
        metadata["effective_kv_seq_len"] = effective_key_seq
        metadata["attention_mode"] = attention_mode
        metadata["visible_pairs_per_head"] = q_seq * (q_seq + 1) // 2 if sparse_mode == 2 else q_seq * k_seq
        metadata["flop_shape_convention"] = (
            "physical_dense_segment" if attention_mode == "exact" else "scaled_compute_proxy"
        )

        _record(
            "aten.matmul.default",
            [q_per_head, k_transposed],
            [scores],
            module_path,
            metadata,
            metadata,
            [q, k],
        )
        _record("aten._softmax.default", [scores], [probabilities], module_path, metadata, metadata)
        _record(
            "aten.matmul.default",
            [probabilities, v_per_head],
            [output],
            module_path,
            metadata,
            metadata,
            [v],
        )

        if attention_mode == "exact":
            output = _per_head_to_bsh(output, batch, head_num, module_path, metadata)

        ctx.save_for_backward(q, k, v)
        ctx.module_path = module_path
        ctx.fused_metadata = metadata
        ctx.head_num = head_num
        ctx.kv_head_num = kv_head_num
        ctx.head_dim = head_dim
        ctx.effective_key_seq = effective_key_seq
        ctx.attention_mode = attention_mode
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        batch, q_seq = q.shape[:2]
        bh = batch * ctx.head_num
        effective_key_seq = ctx.effective_key_seq
        value_head_dim = v.shape[-1] // ctx.kv_head_num
        grads = [_empty_like(tensor) for tensor in (q, k, v)]

        scores = _empty((bh, q_seq, effective_key_seq), q)
        probabilities_t = _empty((bh, effective_key_seq, q_seq), q)
        probabilities = _empty_like(scores)
        grad_output_per_head = _empty((bh, q_seq, value_head_dim), grad_output)
        v_transposed = _empty((bh, value_head_dim, effective_key_seq), v)
        grad_probabilities = _empty_like(scores)
        grad_scores = _empty_like(scores)
        grad_scores_t = _empty((bh, effective_key_seq, q_seq), q)
        k_per_head = _empty((bh, effective_key_seq, ctx.head_dim), k)
        q_per_head = _empty((bh, q_seq, ctx.head_dim), q)
        metadata: dict[str, int | float | str | bool] = ctx.fused_metadata

        # Exact GQA computes one dK/dV contribution per QUERY head, then
        # reduces the groups sharing a KV head. Returning the smaller physical
        # KV gradient directly as a matmul output would undercount its FLOPs.
        exact = ctx.attention_mode == "exact"
        grad_v_matmul = _empty((bh, effective_key_seq, value_head_dim), v) if exact else grads[2]
        grad_k_matmul = _empty((bh, effective_key_seq, ctx.head_dim), k) if exact else grads[1]
        grad_q_matmul = _empty((bh, q_seq, ctx.head_dim), q) if exact else grads[0]

        # dV = P^T @ dO
        _record(
            "aten.matmul.default",
            [probabilities_t, grad_output_per_head],
            [grad_v_matmul],
            ctx.module_path,
            metadata,
            metadata,
            [q, k, v, grad_output],
        )
        if exact:
            grads[2] = _kv_gradient_to_bsh(
                grad_v_matmul, batch, ctx.head_num, ctx.kv_head_num, ctx.module_path, metadata
            )
        # dP = dO @ V^T
        _record(
            "aten.matmul.default",
            [grad_output_per_head, v_transposed],
            [grad_probabilities],
            ctx.module_path,
            metadata,
            metadata,
            [grad_output, v],
        )
        _record(
            "aten._softmax_backward_data.default",
            [grad_probabilities, probabilities],
            [grad_scores],
            ctx.module_path,
            metadata,
            metadata,
        )
        # dQ = dS @ K
        _record(
            "aten.matmul.default",
            [grad_scores, k_per_head],
            [grad_q_matmul],
            ctx.module_path,
            metadata,
            metadata,
            [k],
        )
        if exact:
            grads[0] = _per_head_to_bsh(grad_q_matmul, batch, ctx.head_num, ctx.module_path, metadata)
        # dK = dS^T @ Q
        _record(
            "aten.matmul.default",
            [grad_scores_t, q_per_head],
            [grad_k_matmul],
            ctx.module_path,
            metadata,
            metadata,
            [grad_scores, q],
        )
        if exact:
            grads[1] = _kv_gradient_to_bsh(
                grad_k_matmul, batch, ctx.head_num, ctx.kv_head_num, ctx.module_path, metadata
            )
        return *grads, None, None, None, None, None, None, None, None


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
            self.attention_mode,
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

        if self.attention_mode == "compute_proxy":
            return self._fused_attention(q, k, v, scale=scale, causal=True)
        segments = attention_segments(
            q.shape[1],
            attention_mode=self.attention_mode,
            attention_policy=self.attention_policy,
            block_size=self.block_size,
        )
        outputs = [
            self._fused_attention(
                q[:, start:end],
                k[:, :key_end],
                v[:, :key_end],
                scale=scale,
                causal=causal,
            )
            for start, end, key_end, causal in segments
        ]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


class SimBlockDiffusionAttentionConverter(ModelCustomConverter):
    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if name and type(module) is ScaledCausalSDPA:
                replacement = SimBlockDiffusionAttention(module.converter_config())
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
