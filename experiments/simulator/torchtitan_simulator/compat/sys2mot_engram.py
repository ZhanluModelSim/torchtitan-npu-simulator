# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this tree.

"""Meta contract for the sys2mot Engram multihash n-gram memory boundaries.

The three boundaries reuse the inference-side Engram modeling verbatim
(hashing, table lookup, RMS/SSR gating) and add the training-only gradient
boundaries the reference trains through:

- ``sys2mot_sim.ngram_hash`` reuses the ``NgramHash`` forward model with the
  ``sys2_horner_masked`` variant: the reference's rolling Horner hash
  (``code * 1000003 + shifted + 1) % buckets``) per (n-gram, head) with the
  eligibility mask carried alongside.  Hashing is integer math over ids, so
  no gradient boundary exists.
- ``sys2mot_sim.engram_lookup`` reuses the ``EngramLookup`` forward model:
  hashed keys gather embedding rows.  Its backward scatters the gathered
  gradient back into the table rows (``index_add`` semantics).
- ``sys2mot_sim.engram_gate`` reuses the ``EngramGate`` forward model with
  the ``rms_ssr_residual`` variant: the reference's fp32 dual-RMS normalized
  q/k-weighted dot, sigmoid(sign * sqrt(clamp)) gate over the text-valid
  span, and residual ``hf + gate * value`` expansion.  Its backward supplies
  the four gradients (stream, packed key/value projection, both gate
  weights).

The operators declare capture boundaries, not executable NPU kernels;
real-device calls fail explicitly (the compat install only happens inside
Meta capture sessions).
"""

from typing import Any

import torch

_HASH_VARIANT = "sys2_horner_masked"
_GATE_VARIANT = "rms_ssr_residual"


def _orders_text(orders: tuple[int, ...]) -> str:
    return ",".join(str(order) for order in orders)


@torch.library.custom_op("sys2mot_sim::ngram_hash", mutates_args=())
def ngram_hash(
    ids: torch.Tensor,
    valid: torch.Tensor,
    hash_variant: str,
    heads: int,
    ngram_orders: str,
    output_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise RuntimeError("sys2mot virtual ngram_hash is available only for Meta simulation")


@ngram_hash.register_fake
def _ngram_hash_meta(ids, valid, hash_variant, heads, ngram_orders, output_tokens):
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("ngram hash ids must be a rank-two integer matrix")
    if valid.shape != ids.shape or valid.dtype != torch.bool:
        raise ValueError("ngram hash validity must match the id matrix as bool")
    orders = tuple(int(part) for part in ngram_orders.split(","))
    if not orders or min(orders) < 1 or heads < 1:
        raise ValueError("ngram hash requires positive heads and orders")
    if output_tokens != ids.numel():
        raise ValueError("sys2mot hashes one probe set per id element")
    tokens = output_tokens
    probes = heads * len(orders)
    keys = torch.empty((tokens, probes), dtype=torch.int64, device=ids.device)
    probe_valid = torch.empty((tokens, probes), dtype=torch.bool, device=ids.device)
    return keys, probe_valid


@torch.library.custom_op("sys2mot_sim::engram_lookup", mutates_args=())
def engram_lookup(keys: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("sys2mot virtual engram_lookup is available only for Meta simulation")


@engram_lookup.register_fake
def _engram_lookup_meta(keys, table):
    if keys.ndim != 2 or keys.dtype not in (torch.int32, torch.int64):
        raise ValueError("engram lookup keys must be a rank-two integer matrix")
    if table.ndim != 2 or table.shape[1] <= 0:
        raise ValueError("engram lookup table must be a rank-two row matrix")
    tokens, engram = keys.shape
    if tokens <= 0 or engram <= 0:
        raise ValueError("engram lookup key dimensions must be positive")
    return torch.empty((tokens, engram * table.shape[1]), dtype=table.dtype, device=keys.device)


@torch.library.custom_op("sys2mot_sim::engram_lookup_backward", mutates_args=())
def engram_lookup_backward(
    grad: torch.Tensor, keys: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    raise RuntimeError("sys2mot virtual engram_lookup backward is available only for Meta simulation")


@engram_lookup_backward.register_fake
def _engram_lookup_backward_meta(grad, keys, table):
    if keys.ndim != 2 or grad.ndim != 2 or table.ndim != 2:
        raise ValueError("engram lookup backward requires rank-two operands")
    tokens, engram = keys.shape
    if grad.shape != (tokens, engram * table.shape[1]) or table.shape[1] <= 0:
        raise ValueError("engram lookup backward gradient must match the gathered output")
    return torch.empty_like(table)


def _lookup_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _lookup_backward(ctx, grad):
    keys, table = ctx.saved_tensors
    return (None, engram_lookup_backward(grad, keys, table))


engram_lookup.register_autograd(_lookup_backward, setup_context=_lookup_setup)


@torch.library.custom_op("sys2mot_sim::engram_gate", mutates_args=())
def engram_gate(
    stream: torch.Tensor,
    table: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    mask: torch.Tensor,
    gate_variant: str,
    epsilon: float,
) -> torch.Tensor:
    raise RuntimeError("sys2mot virtual engram_gate is available only for Meta simulation")


@engram_gate.register_fake
def _engram_gate_meta(stream, table, query_weight, key_weight, mask, gate_variant, epsilon):
    if stream.ndim != 3 or min(stream.shape) <= 0:
        raise ValueError("engram gate stream must be a positive (tokens, hc_mult, dim) block")
    tokens, hc_mult, dim = stream.shape
    if stream.dtype != torch.float32 or table.dtype != torch.float32:
        raise ValueError("engram gate math runs in float32 like the reference")
    if table.shape != (tokens, (hc_mult + 1) * dim):
        raise ValueError("engram gate table must pack hc_mult keys and one value")
    if query_weight.shape != (hc_mult, dim) or key_weight.shape != (hc_mult, dim):
        raise ValueError("engram gate weights must be (hc_mult, dim) shaped")
    if query_weight.dtype != torch.float32 or key_weight.dtype != torch.float32:
        raise ValueError("engram gate weights must be float32")
    if mask.shape != (tokens,) or mask.dtype != torch.bool:
        raise ValueError("engram gate mask must be one bool row per token")
    if epsilon <= 0:
        raise ValueError("engram gate epsilon must be positive")
    return torch.empty_like(stream)


@torch.library.custom_op("sys2mot_sim::engram_gate_backward", mutates_args=())
def engram_gate_backward(
    stream: torch.Tensor,
    table: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError("sys2mot virtual engram_gate backward is available only for Meta simulation")


@engram_gate_backward.register_fake
def _engram_gate_backward_meta(stream, table, query_weight, key_weight, grad):
    tokens, hc_mult, dim = stream.shape
    if table.shape != (tokens, (hc_mult + 1) * dim) or grad.shape != stream.shape:
        raise ValueError("engram gate backward operands must match the forward geometry")
    if query_weight.shape != (hc_mult, dim) or key_weight.shape != query_weight.shape:
        raise ValueError("engram gate backward weights must be (hc_mult, dim) shaped")
    stream_grad = torch.empty_like(stream)
    table_grad = torch.empty_like(table)
    query_grad = torch.empty_like(query_weight)
    key_grad = torch.empty_like(key_weight)
    return stream_grad, table_grad, query_grad, key_grad


def _gate_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4])


def _gate_backward(ctx, grad):
    stream, table, query_weight, key_weight = ctx.saved_tensors
    stream_grad, table_grad, query_grad, key_grad = engram_gate_backward(
        stream, table, query_weight, key_weight, grad
    )
    return (stream_grad, table_grad, query_grad, key_grad, None, None, None)


engram_gate.register_autograd(_gate_backward, setup_context=_gate_setup)


def install_sys2mot_engram_meta_support(scope) -> dict[str, str]:
    """Route the sys2mot Engram memory through the fused capture boundaries."""
    import torchtitan_npu.models.sys2mot.model as sys2mot_model

    def engram_forward(self, h: torch.Tensor, ids: torch.Tensor, text_valid: torch.Tensor):
        batch, seq = ids.shape
        orders = tuple(range(2, self.max_ngram + 1))
        keys, probe_valid = ngram_hash(
            ids,
            text_valid,
            "sys2_horner_masked",
            self.heads,
            _orders_text(orders),
            batch * seq,
        )
        masks = probe_valid.view(batch, seq, -1)
        gathered = engram_lookup(keys, self.table.weight)
        emb = gathered.view(batch, seq, -1, self.dim) * masks[..., None]
        packed = self.project(emb.flatten(-2))
        mixed = engram_gate(
            h.float().reshape(batch * seq, self.hc_mult, self.width),
            packed.float().reshape(-1, (self.hc_mult + 1) * self.width),
            self.q_weight.float(),
            self.k_weight.float(),
            text_valid.reshape(-1),
            "rms_ssr_residual",
            self.eps,
        )
        return mixed.reshape(batch, seq, self.hc_mult, self.width).to(h.dtype)

    scope.replace(sys2mot_model.Engram, "forward", engram_forward)
    return {
        "engram_hash_capture_boundary": "sys2mot_sim.ngram_hash[sys2_horner_masked]",
        "engram_lookup_capture_boundary": "sys2mot_sim.engram_lookup",
        "engram_gate_capture_boundary": "sys2mot_sim.engram_gate[rms_ssr_residual]",
    }
