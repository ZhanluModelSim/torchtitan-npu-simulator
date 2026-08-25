# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only DeepSeek V3.2 DSA operators for simulator capture."""

from __future__ import annotations

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _empty(
    shape: tuple[int, ...],
    reference: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    capture = get_active_capture()
    kwargs = {
        "dtype": dtype or reference.dtype,
        "device": reference.device,
    }
    if capture is None:
        return torch.empty(shape, **kwargs)
    with capture.suspend_recording():
        return torch.empty(shape, **kwargs)


def _empty_like(tensor: torch.Tensor) -> torch.Tensor:
    return _empty(tuple(tensor.shape), tensor)


def _record(
    name: str,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    module_path: str,
    *,
    attrs: dict[str, int | float | str] | None = None,
) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            name,
            inputs=inputs,
            outputs=outputs,
            module_path=module_path,
            attrs=attrs,
        )


class _SimLightningIndexer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_indexer,
        key_indexer,
        weights,
        sparse_count,
        module_path,
    ):
        del ctx
        sequence_length = int(query_indexer.shape[1])
        effective_count = min(int(sparse_count), sequence_length)
        indices = _empty(
            (int(query_indexer.shape[0]), sequence_length, effective_count),
            query_indexer,
            dtype=torch.int64,
        )
        _record(
            "npu_lightning_indexer",
            [query_indexer, key_indexer, weights],
            [indices],
            module_path,
            attrs={
                "layout_query": "BSND",
                "layout_key": "BSND",
                "sparse_count": effective_count,
                "sparse_mode": 3,
            },
        )
        return indices

    @staticmethod
    def backward(ctx, grad_indices):
        del ctx, grad_indices
        return None, None, None, None, None


class _SimSparseFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        query_rope,
        key_rope,
        sparse_indices,
        scale_value,
        module_path,
    ):
        output = _empty_like(query)
        stats_shape = (*query.shape[:-1], 8)
        softmax_max = _empty(stats_shape, query, dtype=torch.float32)
        softmax_sum = _empty(stats_shape, query, dtype=torch.float32)
        ctx.save_for_backward(query, key, value, query_rope, key_rope)
        ctx.module_path = module_path
        ctx.scale_value = float(scale_value)
        _record(
            "npu_sparse_flash_attention",
            [query, key, value, sparse_indices, query_rope, key_rope],
            [output, softmax_max, softmax_sum],
            module_path,
            attrs={
                "attention_mode": 2,
                "layout_query": "BSND",
                "layout_kv": "BSND",
                "sparse_mode": 3,
                "scale_value": float(scale_value),
            },
        )
        return output, softmax_max, softmax_sum

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_softmax_max,
        grad_softmax_sum,
    ):
        del grad_softmax_max, grad_softmax_sum
        query, key, value, query_rope, key_rope = ctx.saved_tensors
        grads = [
            _empty_like(query),
            _empty_like(key),
            _empty_like(value),
            _empty_like(query_rope),
            _empty_like(key_rope),
        ]
        _record(
            "npu_sparse_flash_attention_grad",
            [query, key, value, query_rope, key_rope, grad_output],
            grads,
            ctx.module_path,
            attrs={
                "attention_mode": 2,
                "layout_query": "BSND",
                "layout_kv": "BSND",
                "sparse_mode": 3,
                "scale_value": ctx.scale_value,
            },
        )
        return (*grads, None, None, None)


class _SimSparseLightningIndexerKLLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_indexer,
        key_indexer,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        module_path,
    ):
        ctx.save_for_backward(
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
        )
        ctx.module_path = module_path
        return _empty((), query, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_loss):
        (
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
        ) = ctx.saved_tensors
        grad_query_indexer = _empty_like(query_indexer)
        grad_key_indexer = _empty_like(key_indexer)
        grad_weights = _empty_like(weights)
        loss = _empty((1,), query, dtype=torch.float32)
        _record(
            "npu_sparse_lightning_indexer_grad_kl_loss",
            [
                query,
                key,
                query_indexer,
                key_indexer,
                weights,
                sparse_indices,
                softmax_max,
                softmax_sum,
                grad_loss,
            ],
            [grad_query_indexer, grad_key_indexer, grad_weights, loss],
            ctx.module_path,
            attrs={"layout": "BSND", "sparse_mode": 3},
        )
        return (
            None,
            None,
            grad_query_indexer,
            grad_key_indexer,
            grad_weights,
            None,
            None,
            None,
            None,
        )


def sim_dsa_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    q_indexer: torch.Tensor | None = None,
    k_indexer: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    end_pos: torch.Tensor | None = None,
    index_topk: int | None = None,
    topk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the V3.2 fused DSA contract without numerical kernel work."""
    del attn_mask, end_pos
    required = {"index_topk": index_topk}
    if topk_indices is None:
        required.update(
            {
                "q_indexer": q_indexer,
                "k_indexer": k_indexer,
                "weights": weights,
            }
        )
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("DeepSeek V3.2 DSA simulation requires " + ", ".join(missing))
    if k.shape[1] != 1 or v.shape[1] != 1:
        raise NotImplementedError("DeepSeek V3.2 DSA simulation supports MLA absorb mode only")

    capture = get_active_capture()
    module_path = ""
    if capture is not None and capture.module_path_tracker is not None:
        module_path = capture.module_path_tracker.current_path()

    q_indexer_tensor = q_indexer
    k_indexer_tensor = k_indexer
    weights_tensor = weights
    if topk_indices is None:
        assert isinstance(q_indexer_tensor, torch.Tensor)
        assert isinstance(k_indexer_tensor, torch.Tensor)
        assert isinstance(weights_tensor, torch.Tensor)
        sparse_indices = _SimLightningIndexer.apply(
            q_indexer_tensor,
            k_indexer_tensor,
            weights_tensor,
            int(index_topk),
            module_path,
        )
    else:
        sparse_indices = topk_indices.to(dtype=torch.int64)

    q_bsnd = q.transpose(1, 2)
    k_bsnd = k.transpose(1, 2)
    v_bsnd = v.transpose(1, 2)
    q_nope, q_rope = torch.split(
        q_bsnd,
        [module.config.kv_lora_rank, module.config.qk_rope_head_dim],
        dim=-1,
    )
    k_nope, k_rope = torch.split(
        k_bsnd,
        [module.config.kv_lora_rank, module.config.qk_rope_head_dim],
        dim=-1,
    )
    scale_value = float(scale if scale is not None else 1.0)
    output, softmax_max, softmax_sum = _SimSparseFlashAttention.apply(
        q_nope,
        k_nope,
        v_bsnd,
        q_rope,
        k_rope,
        sparse_indices,
        scale_value,
        module_path,
    )
    if q_indexer_tensor is None:
        loss = _empty((), q, dtype=torch.float32)
    else:
        assert isinstance(k_indexer_tensor, torch.Tensor)
        assert isinstance(weights_tensor, torch.Tensor)
        loss = _SimSparseLightningIndexerKLLoss.apply(
            q_nope.detach(),
            k_nope.detach(),
            q_indexer_tensor,
            k_indexer_tensor,
            weights_tensor,
            sparse_indices,
            softmax_max,
            softmax_sum,
            module_path,
        )
    module.last_topk_indices = sparse_indices
    return loss, output.transpose(1, 2)
