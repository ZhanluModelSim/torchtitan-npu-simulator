# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this tree.

"""Meta contract for the sys2mot CSA2 retrieval boundary.

``sys2mot_sim.csa_indexer`` folds the reference's per-chunk full-pool
retrieval — the ReLU head-pooled einsum, the chunked-causal + validity +
candidate masking, the two-level block candidate generation on the source
layer and the top-k selection with ``-1`` sentinels — into one boundary per
layer over the full query set. The masked safe scores and the allowed mask
remain explicit outputs so the training-time KL distillation chain keeps its
eager reference math (including the per-chunk loss fold, replicated by
padding to the chunk grid). The paired backward boundary propagates the
score gradient through the ReLU-weighted einsum into the three training
gradients (query, index key, pooling weights).

The operator declares a capture boundary, not an executable NPU kernel;
real-device calls fail explicitly (the compat install only happens inside
Meta capture sessions).
"""

from typing import Optional

import torch

_FLASH_THETA = 160000.0


@torch.library.custom_op("sys2mot_sim::csa_indexer", mutates_args=())
def csa_indexer(
    query: torch.Tensor,
    index_key: torch.Tensor,
    weights: torch.Tensor,
    key_valid: torch.Tensor,
    prev_candidates: Optional[torch.Tensor],
    index_topk: int,
    compression_ratio: int,
    generate_candidates: bool,
    candidate_blocks: int,
    candidate_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError("sys2mot virtual csa_indexer is available only for Meta simulation")


@csa_indexer.register_fake
def _csa_indexer_meta(
    query,
    index_key,
    weights,
    key_valid,
    prev_candidates,
    index_topk,
    compression_ratio,
    generate_candidates,
    candidate_blocks,
    candidate_block_size,
):
    if query.ndim != 4 or query.dtype != torch.float32:
        raise ValueError("csa indexer queries must be a float32 BQHD block")
    if index_key.ndim != 3 or index_key.dtype != torch.float32:
        raise ValueError("csa indexer keys must be a float32 BMD bank")
    batch, queries, heads, dim = query.shape
    keys = index_key.shape[1]
    if index_key.shape[0] != batch or index_key.shape[2] != dim or keys <= 0 or min(query.shape) <= 0:
        raise ValueError("csa indexer query/key geometry disagrees")
    if weights.shape != (batch, queries, heads) or weights.dtype != torch.float32:
        raise ValueError("csa indexer weights must be float32 (batch, queries, heads)")
    if key_valid.shape != (batch, keys) or key_valid.dtype != torch.bool:
        raise ValueError("csa indexer key validity must cover each compressed bank")
    if prev_candidates is not None and (
        prev_candidates.shape != (batch, queries, keys) or prev_candidates.dtype != torch.bool
    ):
        raise ValueError("csa indexer candidate mask must cover the full score pool")
    # The reference clamps both selections to the available pool with
    # ``min(requested, pool)``, so a large config request may exceed the keys.
    if compression_ratio <= 0 or index_topk <= 0:
        raise ValueError("csa indexer selection must request a positive top-k")
    selection = min(index_topk, keys)
    if generate_candidates:
        if candidate_block_size <= 0 or candidate_blocks <= 0:
            raise ValueError("candidate generation requires positive blocks and block size")
    indices = torch.empty((batch, queries, selection), dtype=torch.int64, device=query.device)
    scores = torch.empty((batch, queries, keys), dtype=torch.float32, device=query.device)
    allowed = torch.empty((batch, queries, keys), dtype=torch.bool, device=query.device)
    candidates = torch.empty((batch, queries, keys), dtype=torch.bool, device=query.device)
    return indices, scores, allowed, candidates


@torch.library.custom_op("sys2mot_sim::csa_indexer_backward", mutates_args=())
def csa_indexer_backward(
    query: torch.Tensor,
    index_key: torch.Tensor,
    weights: torch.Tensor,
    key_valid: torch.Tensor,
    prev_candidates: Optional[torch.Tensor],
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError(
        "sys2mot virtual csa_indexer backward is available only for Meta simulation"
    )


@csa_indexer_backward.register_fake
def _csa_indexer_backward_meta(query, index_key, weights, key_valid, prev_candidates, upstream):
    batch, queries, heads, dim = query.shape
    keys = index_key.shape[1]
    if upstream.shape != (batch, queries, keys) or upstream.dtype != torch.float32:
        raise ValueError("csa indexer backward upstream must be the full-pool score gradient")
    if weights.shape != (batch, queries, heads) or weights.dtype != torch.float32:
        raise ValueError("csa indexer backward weights must be float32")
    return (
        torch.empty_like(query),
        torch.empty_like(index_key),
        torch.empty_like(weights),
    )


def _csa_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])


def _csa_backward(ctx, grad_indices, grad_scores, grad_allowed, grad_candidates):
    query, index_key, weights, key_valid, prev_candidates = ctx.saved_tensors
    query_grad, key_grad, weight_grad = csa_indexer_backward(
        query, index_key, weights, key_valid, prev_candidates, grad_scores
    )
    return (query_grad, key_grad, weight_grad, None, None, None, None, None, None, None)


csa_indexer.register_autograd(_csa_backward, setup_context=_csa_setup)


def install_sys2mot_csa_meta_support(scope) -> dict[str, str]:
    """Route the sys2mot CSA2 retrieval through the fused capture boundary."""
    import math

    import torch.nn.functional as functional

    import torchtitan_npu.models.sys2mot.model as sys2mot_model

    rotary = sys2mot_model.rotary

    def indexer_forward(self, x: torch.Tensor, qr, main_q, valid, runtime, layer: int):
        batch, length = x.shape[0], x.shape[1]
        latent = runtime.latent
        assert latent is not None, "indexer runs on compressed Full Mode KV"
        ratio, keys = runtime.ratio, latent.shape[1]
        if self.owns_k:
            ik = self.norm(self.key(latent))
            runtime.index_key = rotary(
                ik[:, :, None],
                torch.arange(keys, device=x.device) * ratio,
                self.rope_dim,
                _FLASH_THETA,
            )[:, :, 0]
        iq = self.q(qr).view(batch, length, self.index_heads, self.index_dim)
        iq = rotary(iq, torch.arange(length, device=x.device), self.rope_dim, _FLASH_THETA)
        weights = self.weights(x).float() / math.sqrt(self.index_dim * self.index_heads)
        index_key = runtime.index_key
        assert index_key is not None, "index keys are owned or carried from a previous layer"
        latent_valid = runtime.latent_valid
        assert latent_valid is not None, "compressed validity is set by the KV source layer"
        generate = layer == self.candidate_source
        previous = runtime.candidates if layer > self.candidate_source else None
        indices, scores, allowed, candidates = csa_indexer(
            iq.float(),
            index_key.float(),
            weights,
            latent_valid,
            previous,
            self.index_topk,
            ratio,
            generate,
            self.candidate_blocks,
            self.candidate_block_size,
        )
        if generate:
            runtime.candidates = candidates
        runtime.indices = indices
        if not self.training:
            return x.new_zeros(())
        rotated = runtime.rotated
        assert rotated is not None, "distillation target reads the rotated Full Mode KV"
        with torch.no_grad():
            target = torch.einsum("bqhd,bmd->bqhm", main_q.float(), rotated.float())
            target = target.mean(2) / math.sqrt(self.head_dim)
            target = target.masked_fill(~allowed, -1e9).softmax(-1) * allowed
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
        lp = scores.log_softmax(-1)
        kl = (target * (target.clamp_min(1e-9).log() - lp)).sum(-1)
        active = allowed.any(-1) & valid
        # The reference folds the KL loss per query chunk and averages the
        # chunk scalars; zero-pad to the chunk grid to fold identically.
        chunks = -(-length // self.query_chunk)
        pad = chunks * self.query_chunk - length
        per_chunk = functional.pad(kl * active, (0, pad)).view(
            batch, chunks, self.query_chunk
        ).sum((0, 2))
        counts = functional.pad(active, (0, pad), value=False).view(
            batch, chunks, self.query_chunk
        ).sum((0, 2)).clamp_min(1)
        return (per_chunk / counts).mean()

    scope.replace(sys2mot_model.Indexer, "forward", indexer_forward)
    return {
        "csa_indexer_capture_boundary": "sys2mot_sim.csa_indexer",
        "csa_indexer_backward_capture_boundary": "sys2mot_sim.csa_indexer_backward",
    }
