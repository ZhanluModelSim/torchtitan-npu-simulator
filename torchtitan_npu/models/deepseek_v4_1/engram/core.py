# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure PyTorch Engram training reference for DeepSeek-V4.1.

The implementation follows DeepSeek's public Engram demo: compressed token
IDs are hashed with a multiplicative-XOR hash, one row is retrieved from every
N-gram/head table, and the concatenated memory is fused independently into
each mHC branch.  The table lookup is expert-parallel when an EP mesh is
present; all remaining math intentionally stays as ordinary PyTorch so this
module can serve as the numerical reference for a future fused implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.models.base.module import Module


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class EngramTable(Module):
    """Shared hashing and metadata for Host Engram table implementations."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int
        layer_id: int
        ngram_orders: tuple[int, ...]
        num_heads: int
        head_vocab_sizes: tuple[int, ...]
        embedding_dim: int
        # Physical rows, including unreachable tail padding. Keeping this in
        # the flavor config makes checkpoint shape independent of runtime EP.
        num_embeddings: int
        pad_id: int = 2
        hash_seed: int = 0
        tokenizer_path: str | None = None
        token_id_map_path: str | None = None
        # Matches EngramArgs: a table compresses the tokenizer unless the
        # caller opts out. The two defaults must agree, because disagreeing
        # ones let a directly built config train on raw token IDs, which
        # silently changes the compressed vocabulary and therefore every row ID.
        require_token_id_map: bool = True
        compressed_vocab_size: int | None = None

    def __init__(self, config: Config):
        super().__init__()
        if not config.ngram_orders:
            raise ValueError("Engram requires at least one N-gram order.")
        if any(order < 1 for order in config.ngram_orders):
            raise ValueError(f"Engram N-gram orders must be positive, got {config.ngram_orders}.")
        expected_num_heads = len(config.ngram_orders) * config.num_heads
        if len(config.head_vocab_sizes) != expected_num_heads:
            raise ValueError(
                "head_vocab_sizes must contain one size per N-gram/head table: "
                f"expected {expected_num_heads}, got {len(config.head_vocab_sizes)}."
            )
        if config.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {config.embedding_dim}.")
        if not 0 <= config.pad_id < config.vocab_size:
            raise ValueError(f"pad_id ({config.pad_id}) must be in [0, {config.vocab_size}).")

        self.vocab_size = config.vocab_size
        self.layer_id = config.layer_id
        self.ngram_orders = config.ngram_orders
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.pad_id = config.pad_id
        self.hash_seed = config.hash_seed
        self.tokenizer_path = config.tokenizer_path
        self.token_id_map_path = config.token_id_map_path
        self.require_token_id_map = config.require_token_id_map
        self.expected_compressed_vocab_size = config.compressed_vocab_size
        self._head_vocab_sizes = config.head_vocab_sizes

        offsets = [0]
        for size in config.head_vocab_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.logical_num_embeddings = sum(config.head_vocab_sizes)
        self.num_embeddings = config.num_embeddings
        if self.num_embeddings < self.logical_num_embeddings:
            raise ValueError(
                "Engram physical table size cannot be smaller than the hash address space: "
                f"{self.num_embeddings} < {self.logical_num_embeddings}."
            )
        self.weight = torch.nn.Parameter(torch.empty(self.num_embeddings, config.embedding_dim))

        # All of these follow from the flavor: the primes and their offsets from
        # the declared table sizes, the multipliers from the layer's RNG stream,
        # the map from the tokenizer file. Keeping them out of the state dict
        # means a checkpoint carries only what training learned, and lets one
        # written by the reference implementation load as-is.
        self.register_buffer("token_id_map", torch.empty(config.vocab_size, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "head_vocab_sizes",
            torch.tensor(config.head_vocab_sizes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "hash_multipliers",
            torch.empty(max(config.ngram_orders), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("compressed_pad_id", torch.empty((), dtype=torch.int64), persistent=False)

        self.ep_mesh = None

    def _load_token_id_map(self) -> np.ndarray:
        path = self.token_id_map_path
        if path is None and self.tokenizer_path is not None:
            from tokenizers import Tokenizer

            from .token_map import build_token_id_map

            mapping = build_token_id_map(Tokenizer.from_file(self.tokenizer_path))
        elif path is None:
            if self.require_token_id_map:
                raise ValueError(
                    "Tokenizer compression is required for this Engram config, but token_id_map_path is not set."
                )
            return np.arange(self.vocab_size, dtype=np.int64)
        elif not os.path.isfile(path):
            raise FileNotFoundError(
                f"Engram token compression map does not exist: {path}. "
                "Generate it with scripts/generate_engram_token_map.py."
            )
        else:
            mapping = np.load(path, allow_pickle=False)
        if mapping.shape != (self.vocab_size,):
            raise ValueError(f"Engram token compression map must have shape ({self.vocab_size},), got {mapping.shape}.")
        if not np.issubdtype(mapping.dtype, np.integer):
            raise ValueError(f"Engram token compression map must be integral, got {mapping.dtype}.")
        mapping = mapping.astype(np.int64, copy=False)
        if mapping.size and mapping.min() < 0:
            raise ValueError("Engram token compression map contains a negative ID.")
        return mapping

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.weight.device
        token_id_map = self._load_token_id_map()
        compressed_vocab_size = int(token_id_map.max()) + 1
        if compressed_vocab_size <= 0:
            raise ValueError("Engram token compression map is empty.")
        expected = self.expected_compressed_vocab_size
        if expected is not None and compressed_vocab_size != expected:
            # The compressed size bounds the hash multipliers below, so a map
            # from a different tokenizer would change every row ID instead of
            # failing.
            raise ValueError(
                f"Engram token compression map yields a compressed vocabulary of "
                f"{compressed_vocab_size}, but this configuration expects {expected}. "
                f"Regenerate the map from the matching tokenizer with "
                f"scripts/generate_engram_token_map.py."
            )

        # Match the official NumPy demo exactly: each layer has a deterministic
        # RNG stream and uses odd multipliers bounded to avoid int64 overflow
        # for in-range compressed token IDs.
        max_int64 = np.iinfo(np.int64).max
        max_multiplier = max_int64 // compressed_vocab_size
        half_bound = max(1, max_multiplier // 2)
        generator = np.random.default_rng(self.hash_seed + 10007 * self.layer_id)
        random_values = generator.integers(
            low=0,
            high=half_bound,
            size=(max(self.ngram_orders),),
            dtype=np.int64,
        )
        hash_multipliers = random_values * 2 + 1

        offsets = [0]
        for size in self._head_vocab_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.token_id_map = torch.from_numpy(token_id_map.copy()).to(device=device)
        self.head_vocab_sizes = torch.tensor(self._head_vocab_sizes, dtype=torch.int64, device=device)
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        self.hash_multipliers = torch.from_numpy(hash_multipliers).to(device=device)
        self.compressed_pad_id = self.token_id_map[self.pad_id].clone()

    def _parallelize(self, parallelism_context) -> None:
        self.ep_mesh = parallelism_context.get_optional_mesh("ep")
        super()._parallelize(parallelism_context)

    def _shift_tokens(
        self,
        token_ids_T: torch.Tensor,
        positions_T: torch.Tensor | None,
        distance: int,
        token_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if distance == 0:
            return token_ids_T
        compressed_pad_id = _local_tensor(self.compressed_pad_id)
        if token_history is not None and token_history.ndim == 2:
            shifted = token_history[:, distance - 1]
        else:
            shifted = compressed_pad_id.expand_as(token_ids_T).clone()
            if token_history is None:
                shifted[distance:] = token_ids_T[:-distance]
            else:
                source = torch.cat((token_history, token_ids_T), dim=0)
                width = token_history.shape[0]
                start = max(0, distance - width)
                if distance < source.shape[0]:
                    shifted[start:] = source[max(0, width - distance) : source.shape[0] - distance]
        if positions_T is not None:
            shifted = torch.where(
                positions_T >= distance,
                shifted,
                compressed_pad_id,
            )
        return shifted

    def hash(
        self,
        input_ids_T: torch.Tensor,
        positions_T: torch.Tensor | None = None,
        *,
        image_mask: torch.Tensor | None = None,
        token_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return global row IDs with shape ``(T, orders * heads)``."""
        if positions_T is not None and positions_T.shape != input_ids_T.shape:
            raise ValueError(
                "Engram positions must have the same shape as input_ids: "
                f"{tuple(positions_T.shape)} vs {tuple(input_ids_T.shape)}."
            )

        token_id_map = _local_tensor(self.token_id_map)
        compressed_ids_T = token_id_map[input_ids_T.long()]
        if token_history is not None:
            if token_history.ndim not in (1, 2):
                raise ValueError("Engram token history must be a window or a per-token history.")
            if token_history.ndim == 2 and token_history.shape[0] != input_ids_T.shape[0]:
                raise ValueError("Engram per-token history must have one entry per input token.")
            if positions_T is None:
                raise ValueError("Engram token history requires document-relative positions.")
            if image_mask is not None:
                raise ValueError("Engram context parallelism currently supports text-only inputs.")
            token_history = token_id_map[token_history.long()]
        shifts = [
            self._shift_tokens(compressed_ids_T, positions_T, distance, token_history)
            for distance in range(max(self.ngram_orders))
        ]
        if image_mask is not None:
            if image_mask.shape != input_ids_T.shape or image_mask.dtype != torch.bool:
                raise ValueError("Engram image_mask must be boolean and match input_ids")
            # No n-gram may cross an image token, including a multi-token image span.
            alive = ~image_mask
            for distance in range(len(shifts)):
                if distance:
                    previous = torch.zeros_like(alive)
                    previous[distance:] = ~image_mask[:-distance]
                    alive = alive & previous
                shifts[distance] = torch.where(alive, shifts[distance], _local_tensor(self.compressed_pad_id))
        multipliers = _local_tensor(self.hash_multipliers)
        head_vocab_sizes = _local_tensor(self.head_vocab_sizes)
        offsets = _local_tensor(self.offsets)

        hashes = []
        table_idx = 0
        for order in self.ngram_orders:
            mixed = shifts[0] * multipliers[0]
            for distance in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifts[distance] * multipliers[distance])
            end = table_idx + self.num_heads
            moduli = head_vocab_sizes[table_idx:end]
            head_offsets = offsets[table_idx:end]
            hashes.append(torch.remainder(mixed.unsqueeze(-1), moduli) + head_offsets)
            table_idx = end
        return torch.cat(hashes, dim=-1)

    def _distributed_lookup(self, row_ids_N: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Engram table subclasses must implement lookup and gradient accumulation.")

    def forward(
        self,
        input_ids_T: torch.Tensor,
        positions_T: torch.Tensor | None = None,
        *,
        image_mask: torch.Tensor | None = None,
        token_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_ids_TH = self.hash(
            input_ids_T,
            positions_T,
            image_mask=image_mask,
            token_history=token_history,
        )
        T, H = row_ids_TH.shape
        rows = self._distributed_lookup(row_ids_TH.reshape(-1))
        return rows.view(T, H * self.embedding_dim)


class EngramContextGate(Module):
    """Branch-specific keys and gates with a value projection shared by mHC.

    Shape suffixes: T tokens, M branches, D hidden, E memory,
    F = M * D flattened keys, and X = F + D projected key/value width.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        memory_dim: int
        num_branches: int = 4
        norm_eps: float = 1e-5
        signed_sqrt_gate: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_branches = config.num_branches
        self.norm_eps = config.norm_eps
        self.signed_sqrt_gate = config.signed_sqrt_gate
        # One projection producing every branch's key followed by the shared
        # value, in that order. Keeping the two in a single matrix matches how
        # the published model stores them, so its weights load without a
        # bespoke reshape, and it is one GEMM instead of a batched matmul plus
        # a linear.
        self.wkv = torch.nn.Parameter(torch.empty(config.hidden_size * (config.num_branches + 1), config.memory_dim))
        self.q_weight = torch.nn.Parameter(torch.empty(config.num_branches, config.hidden_size))
        self.k_weight = torch.nn.Parameter(torch.empty(config.num_branches, config.hidden_size))

    def forward(
        self,
        hidden_states_TMD: torch.Tensor,
        memory_TE: torch.Tensor,
    ) -> torch.Tensor:
        kv_TX = F.linear(memory_TE, _local_tensor(self.wkv))
        keys_TF, values_TD = kv_TX.split([self.num_branches * self.hidden_size, self.hidden_size], dim=-1)
        # The inference reference accumulates the normalized dot product and
        # residual in FP32, with only the projection in the activation dtype.
        keys_TMD = keys_TF.float().unflatten(-1, (self.num_branches, self.hidden_size))
        queries_TMD = hidden_states_TMD.float()
        norm_weight_MD = _local_tensor(self.q_weight).float() * _local_tensor(self.k_weight).float()
        inverse_norm_TM = torch.rsqrt(queries_TMD.square().mean(-1) + self.norm_eps)
        inverse_norm_TM = inverse_norm_TM * torch.rsqrt(keys_TMD.square().mean(-1) + self.norm_eps)
        logits_TM = (queries_TMD * norm_weight_MD * keys_TMD).sum(-1)
        logits_TM = logits_TM * inverse_norm_TM * self.hidden_size**-0.5
        if self.signed_sqrt_gate:
            magnitude = logits_TM.abs().clamp_min(1e-6).sqrt()
            logits_TM = torch.where(logits_TM >= 0, magnitude, -magnitude)
        gates_TM1 = logits_TM.sigmoid().unsqueeze(-1)
        return gates_TM1 * values_TD.float().unsqueeze(-2)


class Engram(Module):
    """Complete Engram residual module operating on all mHC branches."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        table: EngramTable.Config
        gate: EngramContextGate.Config

    def __init__(self, config: Config):
        super().__init__()
        self.table = config.table.build()
        self.gate = config.gate.build()

    def forward(
        self,
        hidden_states_TMD: torch.Tensor,
        input_ids_T: torch.Tensor,
        positions_T: torch.Tensor | None = None,
        *,
        image_mask: torch.Tensor | None = None,
        token_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Host offload deliberately keeps the authoritative table and its
        # SparseAdam state in FP32.  Keep that storage/optimizer contract while
        # preventing the fetched memory from promoting the model's mixed-
        # precision activation stream (and all following fused operators) to
        # FP32.  Autograd casts the gradient back to the table output dtype.
        memory_TE = self.table(
            input_ids_T,
            positions_T,
            image_mask=image_mask,
            token_history=token_history,
        ).to(dtype=hidden_states_TMD.dtype)
        residual = self.gate(hidden_states_TMD, memory_TE)
        if image_mask is not None:
            residual = residual.masked_fill(image_mask[..., None, None], 0)
        return (hidden_states_TMD.float() + residual).to(hidden_states_TMD.dtype)


__all__ = [
    "Engram",
    "EngramContextGate",
    "EngramTable",
]
