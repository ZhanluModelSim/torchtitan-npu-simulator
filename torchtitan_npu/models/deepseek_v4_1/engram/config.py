# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn

from .core import Engram, EngramContextGate
from .host import HostEngramTable

if TYPE_CHECKING:
    from .. import EngramArgs

_ENGRAM_TABLE_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_ENGRAM_GATE_INIT = {
    "wkv": partial(nn.init.trunc_normal_, std=0.02),
    "q_weight": nn.init.ones_,
    "k_weight": nn.init.ones_,
}


def _is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin primality test for unsigned 64-bit values."""
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime

    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        shifts += 1
        exponent //= 2
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        result = pow(base, exponent, value)
        if result in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            result = result * result % value
            if result == value - 1:
                break
        else:
            return False
    return True


def _next_unused_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    seen.add(candidate)
    return candidate


def _make_engram_configs(
    *,
    hidden_size: int,
    hc_mult: int,
    vocab_size: int,
    engram: EngramArgs,
) -> dict[int, Engram.Config]:
    """Build all layer configs together so their prime table sizes are unique."""
    if not engram.layer_ids or tuple(sorted(set(engram.layer_ids))) != engram.layer_ids:
        raise ValueError(f"Engram layer_ids must be non-empty, unique, and increasing, got {engram.layer_ids}.")
    if (
        not engram.ngram_orders
        or tuple(sorted(set(engram.ngram_orders))) != engram.ngram_orders
        or any(order < 2 for order in engram.ngram_orders)
    ):
        raise ValueError(f"Engram N-gram orders must be unique, increasing, and at least 2, got {engram.ngram_orders}.")
    if len(engram.vocab_size_per_ngram) != len(engram.ngram_orders):
        raise ValueError(
            "vocab_size_per_ngram must contain one entry per N-gram order: "
            f"got {len(engram.vocab_size_per_ngram)} for {len(engram.ngram_orders)} orders."
        )
    if engram.num_heads_per_ngram <= 0:
        raise ValueError("Engram num_heads_per_ngram must be positive.")
    if engram.n_embed_per_ngram <= 0:
        raise ValueError("Engram n_embed_per_ngram must be positive.")
    if engram.table_padding_multiple <= 0:
        raise ValueError("Engram table_padding_multiple must be positive.")
    if engram.n_embed_per_ngram % engram.num_heads_per_ngram != 0:
        raise ValueError(
            f"Engram n_embed_per_ngram ({engram.n_embed_per_ngram}) must be "
            f"divisible by num_heads_per_ngram ({engram.num_heads_per_ngram})."
        )
    embedding_dim = engram.n_embed_per_ngram // engram.num_heads_per_ngram
    memory_dim = len(engram.ngram_orders) * engram.n_embed_per_ngram
    seen_primes: set[int] = set()
    configs: dict[int, Engram.Config] = {}
    for layer_id in engram.layer_ids:
        head_vocab_sizes: list[int] = []
        for base_vocab_size in engram.vocab_size_per_ngram:
            if base_vocab_size <= 0:
                raise ValueError(f"Engram table sizes must be positive, got {base_vocab_size}.")
            prime_search_start = base_vocab_size - 1
            for _ in range(engram.num_heads_per_ngram):
                prime = _next_unused_prime(prime_search_start, seen_primes)
                head_vocab_sizes.append(prime)
                prime_search_start = prime

        logical_rows = sum(head_vocab_sizes)
        physical_rows = (
            (logical_rows + engram.table_padding_multiple - 1)
            // engram.table_padding_multiple
            * engram.table_padding_multiple
        )
        configs[layer_id] = Engram.Config(
            table=HostEngramTable.Config(
                vocab_size=vocab_size,
                layer_id=layer_id,
                ngram_orders=engram.ngram_orders,
                num_heads=engram.num_heads_per_ngram,
                head_vocab_sizes=tuple(head_vocab_sizes),
                embedding_dim=embedding_dim,
                num_embeddings=physical_rows,
                pad_id=engram.pad_id,
                hash_seed=engram.hash_seed,
                token_id_map_path=engram.token_id_map_path,
                require_token_id_map=engram.require_token_id_map,
                compressed_vocab_size=engram.compressed_vocab_size,
                param_init=_ENGRAM_TABLE_INIT,
            ),
            gate=EngramContextGate.Config(
                hidden_size=hidden_size,
                memory_dim=memory_dim,
                num_branches=hc_mult,
                norm_eps=engram.norm_eps,
                param_init=_ENGRAM_GATE_INIT,
            ),
        )
    return configs
