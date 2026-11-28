# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this tree.

"""Meta contract for the sys2mot CSA2 sparse attention boundary.

``sys2mot_sim.window_topk_attention`` folds the reference's per-chunk sparse
attention — the sliding-window local gather, the indexer-selected global
gather with ``-1`` sentinels, the scaled fp32 score einsum, the exact
masked-fill, the sink-column softmax (one extra softmax slot whose
probability is dropped) and the value-weighted output — into one boundary
per layer over the full query set. The chunk loop is a memory tiling of the
same per-query math, so folding it is exact. The paired backward boundary
propagates the output gradient into the query, the joint local+compressed KV
bank (both roles accumulate into one buffer, exactly like the reference's
single ``all_kv`` gather source) and the attention sink parameter.

The operator declares a capture boundary, not an executable NPU kernel;
real-device calls fail explicitly (the compat install only happens inside
Meta capture sessions).
"""

from typing import Optional

import torch


@torch.library.custom_op("sys2mot_sim::window_topk_attention", mutates_args=())
def window_topk_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: Optional[torch.Tensor],
    valid: torch.Tensor,
    sink: torch.Tensor,
    window: int,
    output_dtype: str,
) -> torch.Tensor:
    raise RuntimeError(
        "sys2mot virtual window_topk_attention is available only for Meta simulation"
    )


@window_topk_attention.register_fake
def _window_topk_attention_meta(query, kv, indices, valid, sink, window, output_dtype):
    if query.ndim != 4 or query.dtype != torch.float32:
        raise ValueError("window attention queries must be a float32 BLHD block")
    if kv.ndim != 3 or kv.dtype != torch.float32:
        raise ValueError("window attention KV bank must be a float32 BSD matrix")
    batch, length, heads, dim = query.shape
    if kv.shape[0] != batch or kv.shape[2] != dim or kv.shape[1] <= 0 or min(query.shape) <= 0:
        raise ValueError("window attention query/KV geometry disagrees")
    if valid.shape != (batch, length) or valid.dtype != torch.bool:
        raise ValueError("window attention validity must cover each query row")
    if sink.shape != (heads,) or sink.dtype != torch.float32:
        raise ValueError("window attention sink must be one float32 logit per head")
    if window <= 0:
        raise ValueError("window attention window must be positive")
    if output_dtype not in ("bfloat16", "float32", "float16"):
        raise ValueError("window attention output dtype must be a floating-point width")
    if indices is not None:
        if indices.ndim != 3 or indices.dtype != torch.int64:
            raise ValueError("window attention indices must be an int64 BLK matrix")
        if indices.shape != (batch, length, indices.shape[2]):
            raise ValueError("window attention indices must carry one selection per query")
        if indices.shape[2] <= 0 or indices.shape[2] > kv.shape[1]:
            raise ValueError("window attention selection must fit the KV bank")
    return torch.empty(query.shape, dtype=getattr(torch, output_dtype), device=query.device)


@torch.library.custom_op("sys2mot_sim::window_topk_attention_backward", mutates_args=())
def window_topk_attention_backward(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: Optional[torch.Tensor],
    valid: torch.Tensor,
    sink: torch.Tensor,
    window: int,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError(
        "sys2mot virtual window_topk_attention backward is available only for Meta simulation"
    )


@window_topk_attention_backward.register_fake
def _window_topk_attention_backward_meta(query, kv, indices, valid, sink, window, upstream):
    batch, length, heads, dim = query.shape
    if upstream.shape != query.shape or upstream.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("window attention backward upstream must match the query block")
    if sink.shape != (heads,) or sink.dtype != torch.float32:
        raise ValueError("window attention backward sink must be float32 per head")
    if window <= 0:
        raise ValueError("window attention backward window must be positive")
    return (
        torch.empty_like(query),
        torch.empty_like(kv),
        torch.empty_like(sink),
    )


def _wta_setup(ctx, inputs, output):
    ctx.window = inputs[5]
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])


def _wta_backward(ctx, grad):
    query, kv, indices, valid, sink = ctx.saved_tensors
    query_grad, kv_grad, sink_grad = window_topk_attention_backward(
        query, kv, indices, valid, sink, ctx.window, grad
    )
    return (query_grad, kv_grad, None, None, sink_grad, None, None)


window_topk_attention.register_autograd(_wta_backward, setup_context=_wta_setup)


def install_sys2mot_sparse_attention_meta_support(scope) -> dict[str, str]:
    """Route the sys2mot CSA2 sparse attention through the fused boundary."""
    import math

    import torchtitan_npu.models.sys2mot.model as sys2mot_model

    rotary = sys2mot_model.rotary

    def sparse_attention_forward(self, x, valid, runtime, capture_kv: bool = False):
        batch, length = x.shape[0], x.shape[1]
        positions = torch.arange(length, device=x.device)
        theta = 160000.0 if self.ratio else 10000.0
        qr = self.qnorm(self.qa(x))
        q = rotary(
            self.qb(qr).view(batch, length, self.heads, self.head_dim),
            positions,
            self.rope_dim,
            theta,
        )
        local = rotary(
            self.kvnorm(self.kv(x))[:, :, None], positions, self.rope_dim, theta
        )[:, :, 0]
        aux = x.new_zeros(())
        indices = None
        if self.ratio:
            if self.compressor is not None:
                runtime.latent, runtime.latent_valid = self.compressor(x, valid)
                runtime.ratio = self.ratio
                runtime.rotated = rotary(
                    runtime.latent[:, :, None],
                    torch.arange(runtime.latent.shape[1], device=x.device) * self.ratio,
                    self.rope_dim,
                    theta,
                )[:, :, 0]
            if runtime.ratio != self.ratio:
                raise RuntimeError("Missing Full Mode KV source for this ratio")
            rotated = runtime.rotated
            assert rotated is not None, "Full Mode attends the rotated compressed KV"
            if self.indexer is not None:
                aux = self.indexer(x, qr, q, valid, runtime, self.layer)
            indices = runtime.indices
            assert indices is not None, "Full Mode gathers the indexer-selected keys"
            all_kv = torch.cat((local, rotated), 1)
        else:
            all_kv = local
        y = window_topk_attention(
            q.float(),
            all_kv.float(),
            indices,
            valid,
            self.sink.float(),
            self.window,
            str(x.dtype).removeprefix("torch."),
        )
        y = rotary(y, positions, self.rope_dim, theta, inverse=True)
        grouped = y.reshape(batch, length, self.o_groups, -1)
        y = torch.einsum("blgd,grd->blgr", grouped, self.oa.to(x.dtype))
        result = self.ob(y.flatten(2)) * valid[..., None]
        return (result, aux, local) if capture_kv else (result, aux)

    scope.replace(sys2mot_model.SparseAttention, "forward", sparse_attention_forward)
    return {
        "sparse_attention_capture_boundary": "sys2mot_sim.window_topk_attention",
        "sparse_attention_backward_capture_boundary": (
            "sys2mot_sim.window_topk_attention_backward"
        ),
    }
