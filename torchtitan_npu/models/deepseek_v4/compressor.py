# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.rope import SingleComplexRoPE

from .packed import (
    DSV4PackedMetadata,
    compact_compressed_tensor,
    compact_token_tensor,
    restore_token_tensor,
)


def _assert_spmd_replicated_activation(tensor):
    if get_spmd_backend() == "spmd_types":
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.R},
        )


class Compressor(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        single_rope: SingleComplexRoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        compress_ratio: int = 4
        norm_eps: float = 1e-6

        wkv: Linear.Config
        wgate: Linear.Config
        norm: RMSNorm.Config
        ape: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.single_rope = cfg.single_rope.build()
        self.wkv = cfg.wkv.build()
        self.wkv.weight.data = self.wkv.weight.data.float()
        self.wgate = cfg.wgate.build()
        self.wgate.weight.data = self.wgate.weight.data.float()
        self.norm = cfg.norm.build()
        self.ape = cfg.ape.build().float()

    def forward(self, x, positions=None, attention_masks=None):
        """Compress complete blocks without crossing packed sequence boundaries."""

        if not isinstance(attention_masks, DSV4PackedMetadata):
            raise TypeError(
                "DSV4 compression requires DSV4PackedMetadata attention masks, "
                f"got {type(attention_masks)}."
            )
        metadata = attention_masks
        batch_size, seqlen, _ = x.shape
        ratio = self.compress_ratio
        compressed = metadata.compression_for_ratio(ratio)
        storage_len = seqlen // ratio
        dtype = x.dtype
        rd = self.rope_head_dim

        compact_x = x.reshape(-1, x.shape[-1])[metadata.token_indices].float()
        projected_kv = F.linear(compact_x, self.wkv.weight.float())
        projected_score = F.linear(compact_x, self.wgate.weight.float())
        num_blocks = compressed.block_starts.numel()
        if num_blocks == 0:
            return x.new_zeros((batch_size, storage_len, self.head_dim))

        token_offsets = torch.arange(ratio, device=x.device)
        block_tokens = compressed.block_starts.unsqueeze(1) + token_offsets
        kv = projected_kv[block_tokens]
        score = projected_score[block_tokens] + self.ape.weight.float()

        if self.overlap:
            # Pool the previous same-document block's first-D projection with
            # the current block's second-D projection. At document boundaries,
            # zero/-inf slots prevent the missing predecessor from contributing.
            head_dim = self.head_dim
            overlap_kv = kv.new_zeros((num_blocks, 2 * ratio, head_dim))
            overlap_score = score.new_full(
                (num_blocks, 2 * ratio, head_dim), float("-inf")
            )
            overlap_kv[:, ratio:] = kv[..., head_dim:]
            overlap_score[:, ratio:] = score[..., head_dim:]
            same_document_prev = torch.zeros(
                num_blocks, dtype=torch.bool, device=x.device
            )
            same_document_prev[1:] = (
                compressed.document_ids[1:] == compressed.document_ids[:-1]
            )
            previous = (torch.arange(num_blocks, device=x.device) - 1).clamp_min(0)
            previous_valid = same_document_prev.view(-1, 1, 1)
            overlap_kv[:, :ratio] = torch.where(
                previous_valid,
                kv[previous, :, :head_dim],
                overlap_kv[:, :ratio],
            )
            overlap_score[:, :ratio] = torch.where(
                previous_valid,
                score[previous, :, :head_dim],
                overlap_score[:, :ratio],
            )
            kv, score = overlap_kv, overlap_score

        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(dtype))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = (
            self.single_rope(
                kv_rope.unsqueeze(0).unsqueeze(2),
                compressed.block_positions.unsqueeze(0),
            )
            .squeeze(0)
            .squeeze(1)
        )
        canonical = torch.cat([kv_nope, kv_rope], dim=-1)

        storage = canonical.new_zeros((batch_size * storage_len, self.head_dim))
        storage.index_copy_(0, compressed.storage_indices, canonical)
        storage = storage.view(batch_size, storage_len, self.head_dim)
        _assert_spmd_replicated_activation(storage)
        return storage


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        single_rope: SingleComplexRoPE.Config
        num_index_heads: int = 64
        index_head_dim: int = 128
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        compress_ratio: int = 4
        norm_eps: float = 1e-6

        wq_b: Linear.Config
        weights_proj: Linear.Config
        compressor: Compressor.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.dim = cfg.dim
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.compress_ratio = cfg.compress_ratio
        self.single_rope = cfg.single_rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()

    @staticmethod
    def _rotate_activation(x):
        dtype = x.dtype
        y = x.float()
        dim = y.size(-1)
        assert dim & (dim - 1) == 0, f"Hadamard dim must be a power of two, got {dim}"
        h = 1
        while h < dim:
            y = y.reshape(*y.shape[:-1], -1, 2 * h)
            a, b = y.split(h, dim=-1)
            # Keep each Hadamard butterfly as contiguous half-slices. A size-2
            # pair axis produces ``ModularIndexing(2*i+1, ...)``, which the
            # current torch_npu Inductor scheduler cannot reconstruct.
            y = torch.cat((a + b, a - b), dim=-1)
            y = y.reshape(*y.shape[:-2], dim)
            h *= 2
        return (y * (dim**-0.5)).to(dtype)

    def forward(
        self,
        x,
        qr,
        *,
        positions=None,
        attention_masks=None,
    ):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
            _assert_spmd_replicated_activation(q)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.single_rope(q_rope, positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        _assert_spmd_replicated_activation(q)
        q = self._rotate_activation(q)
        k = self.compressor(
            x,
            positions=positions,
            attention_masks=attention_masks,
        )
        k = self._rotate_activation(k)
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.num_index_heads**-0.5
        )
        return q, k, weights


class IndexSelection(Module):
    """Select compressed-KV candidates from indexer projections.

    Projection and selection are separate configurable components so hardware
    backends can replace only the top-k implementation, such as CANN
    LightningIndexer, without duplicating model projections.

    Packed metadata supplies document and causal boundaries for every query.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int = 512
        compress_ratio: int = 4

    def __init__(self, config: Config):
        super().__init__()
        self.index_topk = config.index_topk
        self.compress_ratio = config.compress_ratio

    def forward(
        self,
        q,
        k,
        weights,
        *,
        attention_masks=None,
    ):
        """Reference top-k over sequence-local complete compressed blocks."""

        if not isinstance(attention_masks, DSV4PackedMetadata):
            raise TypeError(
                "DSV4 index selection requires DSV4PackedMetadata attention "
                f"masks, got {type(attention_masks)}."
            )
        metadata = attention_masks

        compressed = metadata.compression_for_ratio(self.compress_ratio)
        q = compact_token_tensor(q, metadata)
        k = compact_compressed_tensor(k, metadata, self.compress_ratio)
        weights = compact_token_tensor(weights, metadata)
        total_tokens = q.shape[0]
        total_blocks = k.shape[0]
        index_score = torch.einsum("thd,ud->thu", q.float(), k.float())
        index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(dim=1)

        query_docs = metadata.token_sequence_ids
        causal_limit = torch.div(
            metadata.token_positions + 1,
            self.compress_ratio,
            rounding_mode="floor",
        )
        block_docs = compressed.document_ids
        if total_blocks:
            block_local = compressed.block_positions // self.compress_ratio
            valid = query_docs.unsqueeze(1) == block_docs.unsqueeze(0)
            valid = valid & (block_local.unsqueeze(0) < causal_limit.unsqueeze(1))
            index_score = index_score.masked_fill(
                ~valid, torch.finfo(index_score.dtype).min
            )
        else:
            valid = torch.zeros((total_tokens, 0), dtype=torch.bool, device=q.device)

        selected_width = min(self.index_topk, total_blocks)
        selected_score, selected_global = index_score.topk(selected_width, dim=-1)
        selected_valid = torch.gather(valid, dim=-1, index=selected_global)
        compressed_doc_starts = torch.searchsorted(block_docs, query_docs)
        selected_local = selected_global - compressed_doc_starts.unsqueeze(1)
        selected_local = torch.where(selected_valid, selected_local, -1)
        selected_score = selected_score.masked_fill(~selected_valid, float("-inf"))

        pad = self.index_topk - selected_width
        if pad:
            selected_local = F.pad(selected_local, (0, pad), value=-1)
            selected_score = F.pad(selected_score, (0, pad), value=float("-inf"))
        indices = restore_token_tensor(selected_local, metadata, fill_value=-1)
        scores = restore_token_tensor(
            selected_score, metadata, fill_value=float("-inf")
        )
        _assert_spmd_replicated_activation(indices)
        _assert_spmd_replicated_activation(scores)
        return indices, scores
