# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSA2 attention for DeepSeek V4.1.

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension,
    H = ``n_heads``, Dk = ``head_dim``, rd = ``rope_head_dim``,
    N = number of compressed KV entries, K = ``index_topk``.

Each query attends to two sources at once, combined into a single masked softmax by
Attention Gym's ``gather_attn``: its own sliding window over the layer's KV
``[B, L, Dk]``, and the ``K`` compressed entries selected by the indexer out of the
shared compressed KV ``[B, N, Dk]``.  A learned per-head sink logit takes part in the
softmax denominator without contributing a value, so a row with no reachable entry
still produces zeros instead of NaN.

Packed documents reach the operator as the row's packed offsets, read off the metadata's
frames: ``cu_seqlens`` keeps a query's sliding window inside its own document and
``cu_seqlens_k`` numbers the sparse pool per document.  The selection is already in that
numbering -- the indexer emits it document-local, which is also what the fused kernels
read -- so nothing is translated here.  ``positions`` is a per-forward argument, not
metadata, and only drives RoPE.

The indexer's teacher belongs to a port, not to this core.  It is the operator's per-head
log-sum-exp of the full softmax (window, selected compressed entries and sink), turned
into the raw marginal mass ``p`` and emitted on ``topk_scores``' gradient, which the
indexer's SLIKG backward consumes to train the indexer.  A query row with no reachable
compressed entry contributes nothing.  Nothing in this file holds a loss.

The projections use the rope module with the site's un-rotated prefix width.
The reference forward owns Attention Gym attention; fused ports replace the complete
inner-attention forward.
"""

from dataclasses import dataclass, field

import torch
from attn_gym.sparse import gather_attn
from torch import nn
from torchtitan.models.common.attention import BaseAttention, create_varlen_metadata_for_document
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor
from .context_parallel import QueryExchange
from .indexer import HierarchicalIndexer
from .model import DeepSeekV41Metadata


class CompressedSparseInnerAttention2(Module):
    """CSA2's sparse core: sliding window plus the selected compressed entries.

    The whole core is Attention Gym's ``gather_attn`` in one call: ``q`` carries
    ``H`` heads while both KV sources carry a single shared head, and ``topk_indices``
    index the compressed KV -- inside each document's own entries -- with ``-1`` for unused
    slots.  ``compress_ratio`` carries the layer's verified ratio (0 window-only, 1 a
    full-resolution second KV stream, 2 the compressed shared KV): it is the grid the
    sparse pool is addressed on -- a fused port also needs it to translate the metadata's
    document boundaries into its kernel layout -- so it is never guessed from shapes.

    A port that can produce the indexer's teacher replaces this forward wholesale and
    emits it on ``topk_scores``' gradient; this core has no kernel to make one.  Which port
    is in place is read off the requested overrides, one stage earlier, by
    :func:`torchtitan_npu.models.deepseek_v4_1.model._check_indexer_teacher_pair`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        softmax_scale: float
        # The layer's compression ratio: the grid this core addresses its compressed pool on.
        # Carried explicitly rather than guessed from shapes, and the one piece of *this*
        # layer's geometry at which the row's boundaries are read.
        compress_ratio: int

        @property
        def attention_metadata_key(self):
            """This geometry's entry: the backend's class plus its compression ratio.

            One entry per ratio, because ``cu_seq_k`` is ratio-dependent: the builder folds the
            row's boundaries there once, instead of every layer dividing them at call time.  A
            stack that pools at two ratios therefore costs two entries, not a ratio table on
            every layer's config.
            """
            return (type(self), self.compress_ratio)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> DeepSeekV41Metadata:
            """The packed row and this geometry's pool grid, built once per microbatch (#4897).

            Upstream's ``create_varlen_metadata_for_document`` supplies the row's boundaries --
            the "a document starts where ``positions`` resets to 0" convention, with the
            padding-aware and fixed-shape modes this stack does not use yet -- and this backend
            adds the key grid its own ratio implies, so the core never divides a boundary set it
            was handed.  A fused port starts from the same two grids and adds the frames and the
            records only it reads.
            """
            del padding_mask, max_num_documents, max_context_length
            bounds = create_varlen_metadata_for_document(positions).cu_seq_q
            pooled = bounds if self.compress_ratio <= 1 else (bounds // self.compress_ratio).to(torch.int32)
            return DeepSeekV41Metadata(cu_seq_q=bounds, cu_seq_k=pooled, positions_q=positions)

    def __init__(self, config: Config):
        super().__init__()
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.compress_ratio = config.compress_ratio

    def forward(
        self,
        q: torch.Tensor,
        swa_k: torch.Tensor,
        attn_sink: torch.Tensor,
        attention_metadata,
        *,
        cmp_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            q: Queries of shape ``[B, L, H, Dk]``.
            swa_k: Sliding-window KV of shape ``[B, L, Dk]``, shared across heads.
            attn_sink: Per-head sink logits of shape ``[H]``.
            attention_metadata: The forward's varlen metadata; the operator reads the row's
                document boundaries for the sliding-window branch, and grids the compressed
                pool at this layer's own ratio.
            cmp_k: Shared compressed KV of shape ``[B, N, Dk]``.
            topk_indices: Selected compressed entries ``[B, L, K]``, ``-1`` for unused
                slots.  The indexer emits them document-local, which is the numbering the
                operator reads, so they are passed through untouched.
            topk_scores: Accepted for the forward contract and unused here; it exists so
                a port that produces the indexer's teacher has somewhere to carry it.

        Returns:
            Attention output of shape ``[B, L, H, Dk]``.

        This core is the plain reference: it computes attention and nothing else.  The
        indexer's distillation objective is not here -- it needs this operator's softmax
        denominator, and a port that can produce it emits the teacher on ``topk_scores``'
        gradient instead of returning it.
        """
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together.")

        batch, num_tokens, _, head_dim = q.size()
        local_kv_B1LD = swa_k.reshape(batch, 1, num_tokens, head_dim)
        cu_seqlens = attention_metadata.cu_seq_q
        if cmp_k is not None:
            # ``forward`` pairs the two inputs, so ``topk_indices`` is present here.
            assert topk_indices is not None
            sparse_kv_B1ND = cmp_k.reshape(batch, 1, cmp_k.size(1), head_dim)
            assert self.compress_ratio > 0, (
                "Only a layer that pools has a compressed axis, so a core with a compressed "
                f"KV must have a positive ratio (got {self.compress_ratio})."
            )
            kv_indices_BLK = topk_indices
        else:
            # Window-only layer: the sparse pool is empty and every query keeps the
            # window plus the sink.  The two offsets travel together, so the row's own
            # boundaries are passed here and never read: there is no pool to address.
            sparse_kv_B1ND = q.new_zeros(batch, 1, 0, head_dim)
            kv_indices_BLK = torch.empty(batch, num_tokens, 0, dtype=torch.long, device=q.device)
        # The key grid is the metadata's, not this layer's arithmetic: the backend that built the
        # entry folded its ratio there, so the core and the fused port read one set of boundaries.
        cu_seqlens_k = attention_metadata.cu_seq_k

        result = gather_attn(
            q.transpose(1, 2),
            local_kv_B1LD,
            sparse_kv_B1ND,
            kv_indices_BLK,
            attention_sink=attn_sink,
            sliding_window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            scale=self.softmax_scale,
            # TODO: run the fused kernels once they validate this path; ``impl`` is the
            # pinned attn-gym 0.0.16 argument, ``"reference"`` its eager PyTorch path.
            impl="reference",
        )
        assert isinstance(result, torch.Tensor)
        return result.transpose(1, 2)


class Attention(BaseAttention):
    """Latent attention with a grouped output projection, CSA2's per-layer wrapper.

    Projections, both RoPE phases, the compressor and the indexer live here; the sparse
    core is the ``inner_attention`` module.  The shared cross-layer tensors (compressed
    KV, index keys, selected entries, student logits, candidate pool) are threaded
    through as ordinary inputs and returned updated, so a layer's role is visible at the
    call site rather than hidden in mutable state.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        head_dim: int
        # The pinned V3 checkpoint mapping branches on ``q_lora_rank`` (it selects the
        # LoRA-style HF keys), so the field stays even though this module only needs the
        # projection configs.
        q_lora_rank: int
        compress_ratio: int
        inner_attention: CompressedSparseInnerAttention2.Config  # pyrefly: ignore [bad-override]
        rope: RoPE.Config
        compressor: Compressor.Config
        indexer: HierarchicalIndexer.Config
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config
        # The query side's context-parallel movement; it is a module only because the process
        # group it runs on is run-level state, wired by ``parallelize``.
        query_exchange: QueryExchange.Config = field(default_factory=QueryExchange.Config)

    def __init__(self, config: "Attention.Config"):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.compress_ratio = cfg.compress_ratio
        self.rope = cfg.rope.build()
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        # One sink logit per head, fp32 as in the released checkpoint and the kernels.
        self.attn_sink = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        self.compressor = cfg.compressor.build()
        self.indexer = cfg.indexer.build()
        self.inner_attention = cfg.inner_attention.build()
        self.query_exchange = cfg.query_exchange.build()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        *,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(o, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        The returned shared tensors are this layer's contribution to the chain: freshly
        computed where the layer is a source, otherwise the inputs unchanged.
        """
        bsz, seqlen, _ = x.size()
        # Under context parallelism the queries are this rank's chunks while the keys stay the
        # slab it holds: the query side is a permuted copy of the input, and only the
        # projections move -- never the hidden states -- so the keys keep the rows and the
        # positions they always had, and the queries rotate at the ones the metadata carries.
        x_q = self.query_exchange.permute(x.squeeze(0), attention_metadata.exchange).unsqueeze(0)

        qr = self.q_norm(self.wq_a(x_q))
        q = self.wq_b(qr).view(bsz, seqlen, -1, self.head_dim)
        swa_k = self.kv_norm(self.wkv(x))

        # Every layer compresses and indexes: a source publishes new tensors, a reusing
        # layer hands back the ones in flight (asserted inside those modules).
        cmp_k, latent = self.compressor(x, positions, cmp_k)
        # The indexer is trained by distillation alone, so it reads the trunk as
        # constants.  The shared index keys are the exception: a Reindex Mode layer keeps
        # the ``idx_k`` it was handed live, so its consumers go on training the key's
        # owner.
        idx_k, topk_indices, topk_scores, candidates = self.indexer(
            x_q.detach(),
            qr.detach(),
            positions,
            attention_metadata,
            latent=latent.detach() if latent is not None else None,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )

        # The rope config carries the un-rotated prefix width, so the module rotates the
        # trailing span and puts the prefix back.
        q = self.rope(q, positions=attention_metadata.positions_q)
        # The shared KV latent is one rank-2 head; RoPE rotates rank-3 [B, L, N, H].
        swa_k = self.rope(swa_k.unsqueeze(2), positions=positions).squeeze(2)

        uses_cmp = self.compress_ratio > 0
        o = self.inner_attention(
            q,
            swa_k,
            self.attn_sink,
            attention_metadata,
            cmp_k=cmp_k if uses_cmp else None,
            topk_indices=topk_indices if uses_cmp else None,
            topk_scores=topk_scores if uses_cmp else None,
        )
        # Rotate back and project while the rows are still this rank's chunks: every step here is
        # per token, so it commutes with the movement, and ``wo_b``'s output is an order of
        # magnitude narrower than the attention's -- so that is what takes the collective.
        o = self.rope(o, positions=attention_metadata.positions_q, inverse=True)

        # The output projection is grouped: wo_a projects each query-head group on its
        # own, wo_b mixes the per-group results back to the model dimension.  The group
        # count comes from the module so a group-wise sharding can narrow it later.
        o = self.wo_a(o.reshape(bsz * seqlen, self.wo_a.n_batches, -1))
        o = self.wo_b(o.flatten(-2))
        # A permutation, so undoing it is exact and accumulates nothing; ``wo_b`` has already
        # flattened the batch axis, so the row axis is dim 0 here.  Unsharded, the exchange has
        # no plan and hands the tensor straight back.
        o = self.query_exchange.unpermute(o, attention_metadata.exchange)
        return (o, cmp_k, idx_k, topk_indices, topk_scores, candidates)
