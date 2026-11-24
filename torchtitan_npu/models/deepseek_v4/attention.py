# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4 attention projections and sparse attention cores."""

__all__ = [
    "Attention",
    "CompressedSparseAttention",
    "CompressedSparseInnerAttention",
    "DSV4InnerAttention",
    "HeavilyCompressedAttention",
    "SlidingWindowAttention",
    "UnitScaleRMSNorm",
]

from dataclasses import dataclass, field
from typing import Any

import spmd_types as spmd
import torch
import torch_remat as remat
from attn_gym.sparse import AuxRequest, GatherAttnAux, gather_attn
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.lora import _LoRALinearMixin
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v4 import attention as v4_attention
from torchtitan.models.deepseek_v4.attention import (
    DSV4InnerAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
    _assert_spmd_attention_type,
    _packed_cu_seqlens,
)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore [bad-param-name-override]
        # Preserve the original query normalization's input-dtype rounding.
        assert self.eps is not None
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)


class CompressedSparseAttention(DSV4InnerAttention):
    """Gather the document-local slots chosen by the layer's indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k,
        topk_indices,
        topk_scores,
        attn_sink,
        *,
        attention_metadata=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ):
        return self._gather_attn(
            q,
            swa_k,
            cmp_k,
            topk_indices,
            attn_sink,
            _packed_cu_seqlens(attention_metadata),
            topk_scores_TK=topk_scores,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask_T,
        )


class Attention(v4_attention.DeepSeekV4Attention):
    """V4 projections with a configurable Q head norm and separate selection and attention cores."""

    @dataclass(kw_only=True, slots=True)
    class Config(v4_attention.DeepSeekV4Attention.Config):
        q_head_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__(config)
        backend = config.inner_attention
        if config.indexer is not None:
            inner_indexer = self.indexer.inner_indexer
            inner_indexer.return_scores = backend.aux_loss is not None
            if hasattr(backend, "uses_fused_indexer"):
                backend.uses_fused_indexer = inner_indexer.requires_teacher  # pyrefly: ignore [missing-attribute]
                self.inner_attention.uses_fused_indexer = inner_indexer.requires_teacher
            elif inner_indexer.requires_teacher and inner_indexer.return_scores:
                raise ValueError("Fused indexer training requires a sparse attention backend that emits its teacher.")
        self.attention_metadata_key = getattr(backend, "attention_metadata_key", type(self.inner_attention))
        self.q_head_norm = config.q_head_norm.build()

    def project_output_groups(self, grouped):
        """Project each head group with its own base and LoRA B weights."""
        with spmd.local():
            num_groups = grouped.size(1)
            weight = self.wo_a.weight.view(num_groups, self.o_lora_rank, -1)
            if spmd.is_type_checking():
                spmd.assert_type(weight, {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)})
        # ``bmm`` rather than ``einsum``: a Block MX weight is only intercepted by
        # mm / bmm / linear / grouped_mm, so ``einsum`` lowers to unsqueeze+mul on
        # the raw weight and, once FSDP pre-quantizes the shard, dies on
        # ``BlockMXTensor``'s unimplemented ``aten.unsqueeze``.
        output = torch.bmm(grouped.transpose(0, 1), weight.transpose(1, 2)).transpose(0, 1)
        projection = self.wo_a
        if isinstance(projection, _LoRALinearMixin):
            hidden = projection.lora_a(grouped)
            with spmd.local():
                adapter_b = projection.lora_b.weight.view(num_groups, self.o_lora_rank, -1)
                if spmd.is_type_checking():
                    spmd.assert_type(adapter_b, {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)})
            output = output + projection._lora_scaling * torch.einsum("tgl,grl->tgr", hidden, adapter_b)
        return output

    def forward(
        self,
        x,
        attention_metadata=None,
        positions=None,
        aux_loss_denominator=None,
        padding_mask=None,
    ):
        """Apply one DeepSeek V4 attention layer over folded tokens."""
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            # Query, KV, compressor, and indexer branches consume x. Gather
            # once at their common attention boundary.
            x = spmd.redistribute(
                x,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )

        num_tokens = x.size(0)
        rd = self.rope_head_dim

        qr = self.wq_a(x)
        # q_norm reads the wq_a projection output with bare ops.
        remat.recompute_needs_tensor(qr)
        qr = self.q_norm(qr)
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        # The rescale reads the wq_b projection output with bare ops.
        remat.recompute_needs_tensor(q)
        q = self.q_head_norm(q)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        # kv_norm reads the wkv projection output with bare ops.
        remat.recompute_needs_tensor(kv)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cu_seqlens = _packed_cu_seqlens(attention_metadata)
        cmp_k = topk_indices = topk_scores = None
        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            topk_indices, topk_scores = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                cu_seqlens=cu_seqlens,
                attention_metadata=attention_metadata,
            )
        if self.compress_ratio == 4:
            cmp_k = self.compressor(x, positions=positions, cu_seqlens=cu_seqlens)
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(x, positions=positions, cu_seqlens=cu_seqlens)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio == 4:
            o = self.inner_attention(
                q,
                kv,
                cmp_k=cmp_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                attn_sink=attn_sink_param,
                attention_metadata=attention_metadata,
                aux_loss_denominator=aux_loss_denominator,
                padding_mask_T=padding_mask,
            )
        elif self.compress_ratio > 1:
            o = self.inner_attention(
                q,
                kv,
                cmp_k=cmp_k,
                attn_sink=attn_sink_param,
                attention_metadata=attention_metadata,
            )
        else:
            o = self.inner_attention(
                q,
                kv,
                attn_sink_param,
                attention_metadata=attention_metadata,
            )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        with spmd.local():
            n_local_heads = o.shape[1]
            n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
            o = o.view(num_tokens, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
        projected = self.project_output_groups(o)
        with spmd.local():
            projected = projected.reshape(num_tokens, -1)
            _assert_spmd_attention_type(projected, tp=spmd.S(1))
        return self.wo_b(projected)


class CompressedSparseInnerAttention(InnerAttention):
    """Optional NPU sparse-attention backend for DeepSeek-V4.

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
      debugging and numerics reference. The fused backend consumes the same document-local indices and scores.
    - the reference recipe uses ``CompressedSparseAttention`` for CSA and scores distillation
      from its forward LSE. The fused subclass obtains the corresponding teacher from SMLAG.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
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
        def attention_metadata_key(self) -> tuple:
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
        ) -> Any:
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
