# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttention,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor, Indexer
from .metadata import CompressedVarlenMetadata
from .reference import ReferenceCompressedVarlenMetadata
from .token_dispatcher import CPTokenDispatcher


class CompressedSparseInnerAttention(FlexAttention):
    """DeepSeek sparse attention core for DeepSeek-V4 (varlen-typed reference).

    The core attends over the concatenated container KV ``[0, S + n_cmp + 1)``,
    where the first ``S`` positions are the uncompressed sliding-window KV
    (``swa_k``), the next ``n_cmp`` positions are the compressed KV in the
    ``[B, S // ratio, D]`` container grid (``cmp_k``), and the last position is
    a learned attention sink token:

    - sliding window: fixed ``mask_mod`` pattern, restricted to the query
      token's document;
    - compressed blocks: for HCA (``compress_ratio=128``) all causally
      reachable blocks of the same document, also a fixed pattern; for CSA
      (``compress_ratio=4``) each query attends only its top-k selected
      container slots, chosen by ``Indexer.select`` against the dense mask
      from the model's ``build_attention_masks``;
    - attention sink: always attendable via ``score_mod``.

    ``_build_block_mask`` is the single-document container formulation (kept
    for upstream parity and its unit test); ``_build_varlen_block_mask`` is the
    document-packed path driven by ``CompressedVarlenMetadata``.  NPU overrides
    replace the whole ``forward`` (fused SMLA/CSA kernels consume the raw
    ``q / swa_k / cmp_k / idx_q / idx_k / idx_w`` tensors).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(  # pyrefly: ignore [bad-override]
        VarlenAttention.Config
    ):
        # Redeclared as the int DSA window (replaces the inherited varlen
        # ``window_size`` tuple, which is never used by the DSA path).
        window_size: int  # pyrefly: ignore [bad-override]
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        # Consumed by the inherited ``FlexAttention.__init__`` (kernel options
        # for the flex_attention backend of the reference path).
        kernel_options: dict = field(default_factory=dict)

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # pyrefly: ignore [bad-argument-type]
        # Subclasses read ``self.window_size`` as an int.
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.block_size = config.block_size

    def _build_varlen_block_mask(
        self,
        metadata: ReferenceCompressedVarlenMetadata,
        topk_indices: torch.Tensor | None,
        n_cmp: int,
        device,
    ) -> BlockMask:
        """Document-packed block mask driven by ``CompressedVarlenMetadata``.

        The block listing is a superset (window range, selected/full compressed
        region, sink); ``mask_mod`` applies the exact per-token predicates
        (same document, per-document causal limit, top-k selection).
        """
        bsz, seqlen = metadata.batch_size, metadata.seq_len
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        kv_len = seqlen + n_cmp + 1
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq
        sink_idx = seqlen + n_cmp
        ratio = self.compress_ratio
        window_size = self.window_size
        if metadata.plans.get(ratio) is None:
            raise ValueError(f"No compression layout for ratio={ratio}.")
        ref = metadata.reference.ratios[ratio]

        # Static parts (window, sink, HCA range) are hoisted in the metadata;
        # only the CSA top-k blocks are scattered here.
        bm = ref.static_blocks.expand(  # pyrefly: ignore [missing-attribute]
            bsz, 1, -1, -1
        ).clone()
        if topk_indices is not None:
            cmp_block_of = (seqlen + torch.arange(n_cmp, device=device)) // bk
            block_of_topk = cmp_block_of[topk_indices].reshape(
                bsz, n_q_blocks, bq * topk_indices.size(-1)
            )
            bm[:, 0].scatter_add_(
                -1,
                block_of_topk.clamp(0, n_kv_blocks - 1),
                torch.ones_like(block_of_topk, dtype=torch.int32),
            )
        bm = (bm > 0).to(torch.int32)  # pyrefly: ignore [missing-attribute]
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

        cmp_sel = torch.zeros(
            bsz, seqlen, max(n_cmp, 1), dtype=torch.bool, device=device
        )
        if topk_indices is not None:
            cmp_sel.scatter_(2, topk_indices.clamp(0, max(n_cmp, 1) - 1), True)

        doc_of_token = metadata.reference.doc_of_token
        pos_in_doc = metadata.reference.pos_in_doc
        if ratio > 1 and n_cmp > 0:
            cmp_doc = ref.doc_of_block
            cmp_local = ref.block_local
        else:
            # No compressed slots: keep the gather safe with dummy values.
            cmp_doc = torch.full(
                (bsz, max(n_cmp, 1)), -1, dtype=torch.int32, device=device
            )
            cmp_local = torch.full(
                (bsz, max(n_cmp, 1)), -1, dtype=torch.int32, device=device
            )

        def csa_varlen_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            doc_q = doc_of_token[b, q_idx]
            kv_safe = kv_idx.clamp(0, seqlen - 1)
            swa = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < window_size)
                & (doc_of_token[b, kv_safe] == doc_q)
            )
            is_sink = kv_idx == sink_idx
            if ratio > 1:
                c = kv_idx - seqlen
                in_cmp = (c >= 0) & (c < n_cmp)
                c_safe = c.clamp(0, max(n_cmp, 1) - 1)
                same_doc = (
                    cmp_doc[  # pyrefly: ignore [unsupported-operation]
                        b, c_safe
                    ]
                    == doc_q
                )
                causal = cmp_local[  # pyrefly: ignore [unsupported-operation]
                    b, c_safe
                ] < torch.div(pos_in_doc[b, q_idx] + 1, ratio, rounding_mode="floor")
                if topk_indices is not None:
                    topk_sel = cmp_sel[b, q_idx, c_safe]
                    return swa | (in_cmp & same_doc & causal & topk_sel) | is_sink
                return swa | (in_cmp & same_doc & causal) | is_sink
            return swa | is_sink

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=csa_varlen_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attn_sink: torch.Tensor | None = None,
        *,
        attention_masks: ReferenceCompressedVarlenMetadata | None = None,
    ) -> torch.Tensor:
        if not isinstance(attention_masks, CompressedVarlenMetadata):
            raise TypeError(
                "CompressedSparseInnerAttention requires CompressedVarlenMetadata "
                f"attention masks, got {type(attention_masks)}."
            )
        if attn_sink is None:
            raise ValueError("CompressedSparseInnerAttention requires attn_sink")

        metadata = attention_masks
        bsz, seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(1)
        sink_idx = seqlen + n_cmp

        topk_indices = None
        if self.compress_ratio == 4:
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError(
                    "CompressedSparseInnerAttention requires idx_q, idx_k, and "
                    "idx_w when compress_ratio=4"
                )
            if metadata.plans.get(4) is None:
                raise ValueError(
                    "CompressedSparseInnerAttention requires the ratio-4 "
                    "compression layout for indexer selection."
                )
            topk_indices, _ = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                metadata.reference.ratios[  # pyrefly: ignore [bad-argument-type]
                    4
                ].dense_mask,
                self.index_topk,
            )

        kv = swa_k.unsqueeze(2)
        if cmp_k is not None:
            kv = torch.cat([kv, cmp_k.unsqueeze(2)], dim=1)
        sink_kv = kv.new_zeros((bsz, 1, 1, head_dim))
        kv = torch.cat([kv, sink_kv], dim=1)

        block_mask = self._build_varlen_block_mask(
            metadata, topk_indices, n_cmp, q.device
        )

        def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(
                kv_idx == sink_idx,
                attn_sink[h],  # pyrefly: ignore [unsupported-operation]
                score,
            )

        return super().forward(
            q,
            kv,
            kv,
            attention_masks=block_mask,
            score_mod=v4_sink_score_mod,
            scale=self.softmax_scale,
            enable_gqa=True,
        )


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        head_dim: int
        rope_head_dim: int
        q_lora_rank: int
        n_groups: int
        compress_ratio: int
        norm_eps: float

        # Declare submodule configs as fields so sharding can be assigned before
        # the modules are built.
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

        # Built only for ``compress_ratio > 1`` layers (``indexer`` only for
        # ratio-4 CSA layers); the registry passes ``None`` otherwise.
        compressor: Compressor.Config | None
        indexer: Indexer.Config | None

        # The CP token dispatcher (the RoutedExperts mirror): a submodule of
        # the attention, wired once by ``Attention.parallelize``.
        token_dispatcher: CPTokenDispatcher.Config = field(
            default_factory=CPTokenDispatcher.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.rope = cfg.rope.build()

        self.token_dispatcher = cfg.token_dispatcher.build()

        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        # Bare head-wise sink parameter (fp32), matching the inference
        # reference and the kernels' ``[N1]`` sink contract.
        self.attn_sink = torch.nn.Parameter(
            torch.empty(cfg.n_heads, dtype=torch.float32)
        )

        self.compressor = cfg.compressor.build() if cfg.compressor is not None else None
        self.indexer = cfg.indexer.build() if cfg.indexer is not None else None

        self.inner_attention = cfg.inner_attention.build()

    def parallelize(self, parallel_dims) -> None:
        """Parallelize the attention, then wire the CP mesh on the
        attention's own token dispatcher (the ``RoutedExperts.parallelize``
        mirror).  The compressors' dispatchers are wired by their owners'
        ``parallelize`` through the framework's ``Module.parallelize``
        recursion."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(cp_mesh=parallel_dims.get_optional_mesh("cp"))

    def forward(self, x, attention_masks, positions):
        """The unified attention forward (CP and non-CP).

        The Q side and the swa projection run on the local stream; the
        token dispatcher's ops serve every consumer with no context-
        parallel special-casing: ``gather`` exchanges the post-RoPE
        ``swa_k`` rows (the window plan) into the packed ori stream, the
        compressors gather their own block rows internally, and ``select``
        packs the pooled streams into the padded containers.  The
        containers' all-gather is declarative — the core's
        ``ShardingConfig`` (``cp: S(1) -> R``) emits it at the core
        boundary.
        """
        window = attention_masks.window
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, -1, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.rope(q_rope, positions=positions)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # The swa projection + RoPE run on the local rows (the sender's own
        # doc-relative positions — the attention's positions convention
        # resets per document); the window gather exchanges the post-RoPE
        # rows into the packed ori stream.
        swa_k = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(swa_k, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(
            kv_rope.unsqueeze(2),
            positions=positions.reshape(1, -1),
        ).squeeze(2)
        swa_k = torch.cat([kv_nope, kv_rope], dim=-1)
        swa_k = self.token_dispatcher.gather(swa_k, window)

        cmp_k = None
        idx_q = idx_k = idx_w = None

        if self.compress_ratio > 1 and self.indexer is not None:
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_masks=attention_masks,
            )
            # The indexer's outputs: idx_q / idx_w (local), idx_k (the
            # pooled stream — packed into the container).
            idx_k = self.token_dispatcher.select(idx_k, attention_masks.plans[4])

        if self.compress_ratio > 1:
            assert self.compressor is not None, (
                "compress_ratio > 1 requires the compressor submodule."
            )
            plan = attention_masks.plans[self.compress_ratio]
            pooled = self.compressor(x, attention_masks)
            cmp_k = self.token_dispatcher.select(pooled, plan)

        # Inner-attention positional contract: absent components are None.
        #   sink + swa_k always; + cmp_k when compress_ratio > 1;
        #   + idx_q/idx_k/idx_w when compress_ratio == 4 (indexer layer).
        o = self.inner_attention(
            q,
            swa_k,
            cmp_k,
            idx_q,
            idx_k,
            idx_w,
            attn_sink=self.attn_sink,
            attention_masks=attention_masks,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        # ``wo_a`` is a BatchedLinear over the head groups; group the heads
        # before the per-group matmul.
        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        o = self.wo_a(o)
        o = o.reshape(bsz, seqlen, -1)
        return self.wo_b(o)
