# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from attn_gym.sparse import AuxRequest, GatherAttnAux, gather_attn
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor, Indexer, LightningIndexer
from .metadata import DeepSeekV4Metadata, build_v4_metadata


class UnitScaleRMSNorm(RMSNorm):
    """RMSNorm with a fixed unit scale excluded from checkpoints."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        elementwise_affine: bool = field(default=False, init=False)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        del self.weight
        self.register_buffer("weight", torch.ones(config.normalized_shape), persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        self.weight.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the original query normalization's input-dtype rounding.
        assert self.eps is not None
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)


class CompressedSparseInnerAttention(Module):
    """DeepSeek sparse attention core for DeepSeek-V4.

    The core is Attention Gym's ``gather_attn`` over the packed offsets the metadata already
    carries, so no document-aware block mask is built per layer:

    - ``compress_ratio == 1``: sliding window plus sink, no compressed stream;
    - ``compress_ratio == 4`` (CSA): the selector's picks, already numbered inside each query's
      own document -- the numbering ``gather_attn`` addresses a packed pool by;
    - a higher ratio (the model's HCA layers): every causally reachable compressed slot of the
      query's own document, which those layers reach without a selector.

    NPU overrides replace the whole class; they answer the same contract.  Two limits of this
    default are worth knowing before it is used for training:

    - it does **not** chunk the query axis: the operator materialises intermediates that grow
      with the sequence length, the window/selection width and the head dimension, so it is a
      debugging and numerics reference rather than the path to train on -- the fused
      ``sparse_attn.asc`` / ``sparse_attn.pypto`` recipes are the ones the model ships (both
      replace this class and keep working under context parallel, which this core refuses);
    - it carries no indexer loss of its own: the LI objective is a property of a fused port (its
      SMLAG backward hands the fused selector its teacher), so a recipe that runs this core
      trains no gradient for the indexer's compressor.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # TODO: Revisit the upstream attention config contract on the next refactor;
        # reuse or inherit the upstream interface when it can express this CP-capable
        # sparse backend without classifying it as VarlenInnerAttention.
        # The model's window is a count of preceding tokens.
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        return_lse: bool = False
        """Hand back the per-head log-sum-exp beside the output.  A test hook: nothing in the
        model consumes it, and a fused core takes its own from the kernel."""

        @property
        def attention_metadata_key(self):
            """This layer's own entry: the core's class plus its compression ratio.

            One entry per geometry is what lets the builder derive exactly the grid this layer
            addresses -- no stack-wide ratio table on any config, and no entry shared by layers
            whose kernels address different pools.
            """
            return (type(self), self.compress_ratio)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> DeepSeekV4Metadata:
            """This geometry's per-microbatch contract (upstream #4897).

            Built from the pre-shard ``positions`` alone: the entry carries the row's boundaries
            and its own ratio's fold of them, so one build serves every layer of this geometry --
            the decoder keys the map by this config's :attr:`attention_metadata_key`.  This
            builder's contract refuses context parallel; the fused core's config builds an entry
            whose hook rebuilds it for this rank.
            """
            del padding_mask, max_num_documents, max_context_length
            # A document starts where ``positions`` resets to 0; this layer's ratio folds the row
            # into the pool grid its kernels address.  The selector's records are the selector
            # port's business: this builder produces the row's frames only, and a fused selector
            # backend attaches its own records.
            return build_v4_metadata(positions, self.compress_ratio)

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Subclasses read ``self.window_size`` as an int.
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.return_lse = config.return_lse

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        attn_sink: torch.Tensor,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        attention_metadata: DeepSeekV4Metadata,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # ``topk_scores`` is the selector's student-logit edge (or a fused port's teacher
        # carrier).  The reference core only gathers by ``topk_indices``, so it accepts the
        # tensor for the shared forward contract and never reads it.
        del topk_scores
        if not isinstance(attention_metadata, DeepSeekV4Metadata):
            raise TypeError(
                "CompressedSparseInnerAttention requires DeepSeekV4Metadata "
                f"attention metadata, got {type(attention_metadata).__name__}."
            )
        if self.compress_ratio == 4 and (cmp_k is None) != (topk_indices is None):
            raise ValueError("ratio-4 attention requires cmp_k and topk_indices together.")
        seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(0)
        # The frames the kernels grid: the query's and, for a pooling geometry, the compressed
        # pool's.  The reference path and a fused port read the same ones.
        cu_seqlens = attention_metadata.q.cu_seqlens
        if n_cmp == 0:
            # A window-only layer: the pool is empty, and the operator wants both offsets.
            cu_seqlens_k = cu_seqlens
            kv_indices = q.new_empty((1, seqlen, 0), dtype=torch.long)
            sparse_kv = q.new_zeros((1, 1, 0, head_dim))
        else:
            # ``n_cmp`` is this tensor's width, so the frame is present here.
            assert cmp_k is not None
            cmp_frame = attention_metadata.cmp_k
            if cmp_frame is None:
                raise ValueError("a compressed KV needs the entry's compressed frame")
            cu_seqlens_k = cmp_frame.cu_seqlens
            # The selection is an input: the selector's picks at ratio four, and the entry's
            # precomputed reference selection for a geometry without a selector.
            kv_indices = (
                topk_indices.unsqueeze(0)
                if self.compress_ratio == 4 and topk_indices is not None
                else attention_metadata.slots
            )
            if kv_indices is None:
                raise ValueError("the entry carries no selection for this geometry")
            sparse_kv = cmp_k.reshape(1, 1, n_cmp, head_dim)

        aux = AuxRequest(lse=True) if self.return_lse else None
        result = gather_attn(
            q.transpose(0, 1).unsqueeze(0),
            swa_k.reshape(1, 1, seqlen, head_dim),
            sparse_kv,
            kv_indices,
            attention_sink=attn_sink,
            sliding_window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            scale=self.softmax_scale,
            impl="reference",
            return_aux=aux,
        )
        if aux is None:
            assert isinstance(result, torch.Tensor)
            return result.squeeze(0).transpose(0, 1)
        assert isinstance(result, tuple)
        output, gathered = result
        assert isinstance(gathered, GatherAttnAux) and gathered.lse is not None
        return output.squeeze(0).transpose(0, 1), gathered.lse.squeeze(0)


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
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
        q_head_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

        # Built only for ``compress_ratio > 1`` layers (``indexer``/``lightning_indexer``
        # only for ratio-4 CSA layers); the registry passes ``None`` otherwise.
        compressor: Compressor.Config | None
        indexer: Indexer.Config | None
        lightning_indexer: LightningIndexer.Config | None
        # The sparse core the decoder replaces wholesale with ``sparse_attn.asc``.  It is the
        # layer's only metadata backend and the CP boundary for its KV streams; the annotation
        # stays the base type because a port's config is a replacement of this node.
        inner_attention: Module.Config

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

        # The CP mesh this layer restores its key side on, wired by ``parallelize``; ``None`` at
        # one degree, where the metadata carries no plan and nothing moves.
        self.cp_group = None

        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.q_head_norm = cfg.q_head_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        # Bare head-wise sink parameter (fp32), matching the inference
        # reference and the kernels' ``[N1]`` sink contract.
        self.attn_sink = torch.nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))

        self.compressor = cfg.compressor.build() if cfg.compressor is not None else None
        self.indexer = cfg.indexer.build() if cfg.indexer is not None else None
        self.lightning_indexer = cfg.lightning_indexer.build() if cfg.lightning_indexer is not None else None
        self.inner_attention = cfg.inner_attention.build()

    def _parallelize(self, parallelism_context) -> None:
        """Take the CP mesh the layer's key stream is restored on, then recurse.

        The compressor and its carry exchange take the same mesh through their own
        ``parallelize`` (the framework recurses into every child module), so nothing here wraps
        them: their module paths, sharding and checkpoint entries stay as they are.
        """
        self.cp_group = parallelism_context.get_optional_mesh("cp")
        super()._parallelize(parallelism_context)

    def forward(self, x, attention_metadata, positions):
        """The unified attention forward (CP and non-CP).

        The input layout already put these rows on **chunk ``r`` of every document**, so the
        query side projects them as they arrive and never moves.  The key side projects, rotates
        and pools the same rows -- every step is per token, so it commutes with the layout -- and
        each consumer restores what it reads on its own: the compressor its pooling window, the
        core this rank's swa window, whose sharding then gathers the whole row at its boundary.
        """
        num_tokens = x.size(0)
        metadata = attention_metadata

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(num_tokens, -1, self.head_dim)
        q = self.q_head_norm(q)
        # RoPE owns the [nope | rope] split: it receives the full-width
        # tensor and rotates only the trailing rope_head_dim channels.  The
        # positions are the ones the input layout carried with these rows.
        q = self.rope(q, positions=positions)

        # The key stream's own positions are the entry's: the input's at one degree, this rank's
        # window under CP.  The queries rotate at the positions the input carried.
        position_k = metadata.position_k

        # The swa projection + RoPE run on the rows the layout produced: RoPE is per token, so it
        # commutes with the movement.  The restore then hands the core this rank's window, whose
        # sharding gathers the rows to the whole row at the core's own boundary.
        swa_k = self.kv_norm(self.wkv(x))
        swa_k = self.rope(swa_k.unsqueeze(1), positions=positions).squeeze(1)
        swa_k = metadata.key_stream(swa_k, group=self.cp_group)

        cmp_k = None
        idx_q = idx_k = idx_w = None

        if self.compress_ratio > 1 and self.indexer is not None:
            # The selector's queries and weights ride the doc-chunk rows with the attention's;
            # its compressor restores and carries for the keys it pools.
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_metadata=metadata,
            )
        if self.compress_ratio > 1:
            assert self.compressor is not None, "compress_ratio > 1 requires the compressor submodule."
            # The compressor owns the restore and the ratio-4 carry: it pools the window this rank
            # holds and hands back the row's blocks, so the layer never sees either step.
            cmp_k = self.compressor(
                x,
                position_k,
                cu_seqlens=metadata.swa_k.cu_seqlens,
                attention_metadata=metadata,
            )

        # Every selector returns indices and scores: the reference exposes student logits,
        # while the AscendC, PyPTO and quantized backends expose the teacher-gradient carrier.
        topk_indices = topk_scores = None
        if self.lightning_indexer is not None:
            topk_indices, topk_scores = self.lightning_indexer(idx_q, idx_k, idx_w, attention_metadata=metadata)

        # Inner-attention positional contract: sink + swa_k always; the compressed pool and
        # the selector's ``(topk_indices, topk_scores)`` pair only on a pooled layer.
        o = self.inner_attention(
            q,
            swa_k,
            self.attn_sink,
            cmp_k=cmp_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            attention_metadata=metadata,
        )

        o = self.rope(o, positions=positions, inverse=True)

        # ``wo_a`` is a BatchedLinear over the head groups; group the heads
        # before the per-group matmul.
        n_local_groups = self.n_groups // (self.n_heads // o.shape[1])
        o = o.view(num_tokens, n_local_groups, -1)
        o = self.wo_a(o)
        o = o.reshape(num_tokens, -1)
        o = self.wo_b(o)
        # The projection is per token, so ``wo_b``'s output stays in the frame the input
        # arrived in: the residual and every other consumer are token-wise, and the loss is
        # taken against labels the same layout moved.
        return o
