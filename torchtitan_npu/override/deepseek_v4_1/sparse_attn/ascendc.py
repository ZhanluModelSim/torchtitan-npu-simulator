# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 SMLA attention and SMLAG-backed indexer distillation."""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.common.metadata_extension import (
    register_pytree_node_for_dataclass,
)
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.context_parallel import cp_plan
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import (
    build_indexer_metadata,
)
from torchtitan_npu.override.deepseek_v4_1.metadata import (
    AicpuGeometry,
    AicpuMetadata,
    AscLightningIndexerMetadata,
    AscSparseAttentionMetadata,
    KernelFrame,
)
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context


def validate_compressed_frame(ratio: int, cmp_k: KernelFrame | None) -> None:
    """Reject a compressed frame with a partial trailing group.

    Every document must be a whole number of pooling groups, so the compressed axis has no
    partial trailing group and ``cmp_residual_kv`` is zero throughout.  A non-zero value here
    means the loader's document alignment and this layer's ``compress_ratio`` disagree -- the
    kernels would then grid an axis the pooling did not produce -- so it is checked rather than
    passed on.
    """
    if cmp_k is None or cmp_k.residual is None:
        return
    if bool(cmp_k.residual.any()):
        raise ValueError(
            f"compress_ratio {ratio} has a partial trailing group: a document length "
            f"is not a multiple of the pooling ratio ({cmp_k.residual.tolist()}). The "
            "dataloader's document alignment must be a multiple of every ratio the stack pools at."
        )


def kernel_options(
    *,
    ratio: int,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    window_size: int,
):
    """The kernel options, read off the metadata's frames.

    Both the metadata call and the main call take these, and they must agree
    option-for-option; reading one source is what makes that structural rather than a
    convention two call sites have to keep in step.
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_ori_kv=swa_k.cu_seqlens,
        cu_seqlens_cmp_kv=None if cmp_k is None else cmp_k.cu_seqlens,
        # The per-sequence lengths actually addressed -- how an operator is told to read a prefix
        # of its operand.  Under context parallelism they carry the whole query movement:
        # ``seqused_q`` is this rank's chunk and ``seqused_ori_kv`` its reach, so the sliding
        # window and the compressed causal limit are anchored at ``seqused_ori_kv - seqused_q``,
        # this rank's chunk offset inside the document.  At ``cp_size = 1`` each of them equals the
        # corresponding boundary difference, so passing them is a no-op there rather than a second
        # code path.
        seqused_q=q.seqused,
        seqused_ori_kv=swa_k.seqused,
        seqused_cmp_kv=None if cmp_k is None else cmp_k.seqused,
        cmp_residual_kv=None if cmp_k is None else cmp_k.residual,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=4,
        cmp_mask_mode=0 if _IS_A5 and cmp_k is None else 3,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
    )


# ---------------------------------------------------------------------------
# The entry: this geometry's frames and its AI_CPU metadata
# ---------------------------------------------------------------------------
#
# The AI_CPU metadata operators depend only on the frames of one forward and the layer's static
# geometry, never on the operands, so they are computed once per microbatch and read by every layer
# of the geometry instead of being re-run per layer inside the compiled region.  This module builds
# exactly what its own kernels read: the sparse attention's pair and the BF16 selector's pair.  A
# stack that runs a different backend (the quantized attention) builds its own entry there -- the
# `fwd_sparse` / `sparse_length` fields on this entry stay empty until the non-quantized candidate
# kernels land.
# ``..metadata`` describes the structure; this module owns the entry that carries it.


@dataclass(frozen=True, kw_only=True, slots=True)
class AscV41Metadata(DeepSeekV41Metadata):
    """The row, this geometry's kernel frames, and the AI_CPU metadata its kernels read.

    One entry per ``(port, ratio)``: the map key is this port's own config plus the layer's
    compression ratio, so every layer reads an entry built for its own geometry and no builder
    has to know the stack's ratio table.  The frames are this geometry's own -- the query's, the
    sliding window's and the compressed axis's -- and ``aicpu_metadata`` holds the operator
    outputs for exactly those frames, computed at the eager boundary once per microbatch and
    rebuilt per rank under context parallelism from the rank-local frames.

    Consumers read the fields directly (``cmp_k.seqused``, ``aicpu_metadata.sparse_attn.bwd``):
    one model runs one selection width, so there is nothing to look up.  Nothing here compares the
    metadata's geometry with the layer's own config -- that would need the numbers the reference core
    never sees -- so an override stated with the wrong widths surfaces as the kernels' shape error.
    Every field is a tensor, a primitive or another registered node, because the class is a pytree
    node (see the registration below).
    """

    ratio: int
    q: KernelFrame
    swa_k: KernelFrame
    cmp_k: KernelFrame | None
    aicpu_metadata: AicpuMetadata

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        # The seam dispatches on ``type(metadata)``, so this hook's contract is its own class;
        # widening the parameter would accept a plain row and lose this geometry's ratio.
        attention_metadata: "AscV41Metadata",
        *,
        permutation: torch.Tensor | None,
    ) -> "AscV41Metadata":
        """This rank's frames and metadata (upstream #4617's CP half), for one geometry.

        The row and the rank come from the shard context, because this rank's copy is derived
        from all of it.  ``permutation`` is unused and a permuted run is refused: these frames are
        contiguous slabs of every document, and a load balancer's permutation would have to be
        folded into the chunk grid, the query exchange and the engram halo alike.
        """
        del permutation
        context = cp_shard_context()
        if context.cp_mesh is None or context.positions is None:
            raise ValueError("prepare_cp_metadata needs the CP mesh and the pre-shard positions")
        if context.load_balancer_type is not None:
            raise ValueError(
                f"the V4.1 fused stack shards contiguous slabs, but the context-parallel load "
                f"balancer is {context.load_balancer_type!r}: run with "
                "--parallelism.context-parallel-load-balancer None."
            )
        batch_size, row_len = context.positions.shape[0], context.positions.shape[-1]
        return shard_asc_v41_metadata(
            attention_metadata,
            cp_size=context.cp_mesh.size(),
            cp_rank=context.cp_mesh.get_local_rank(),
            batch_size=batch_size,
            seq_len=row_len // context.cp_mesh.size(),
        )


# The registration is asymmetric on purpose, and reserved: DSV4 registers its whole masks object
# because the graph_trainer's ``minimal_fx_tracer`` requires ``attention_masks`` to flatten into
# tensor/primitive leaves, while V4.1 has no tracer path today.  The parent
# ``DeepSeekV41Metadata`` therefore stays an opaque pytree leaf -- the golden path's flattening
# behavior is untouched -- and only the Asc node is registered, mirroring DSV4's seam so the
# fused metadata is pytree-transparent from day one when such a path arrives.
register_pytree_node_for_dataclass(AscV41Metadata)


def build_fused_metadata(row: DeepSeekV41Metadata, *, ratio: int, geometry: AicpuGeometry) -> AscV41Metadata:
    """This layer's geometry, built from the row: frames plus the AI_CPU metadata it reads.

    The row is the only input: ``cu_seq_k`` is already folded at this ratio by the reference
    backend, so the compressed frame needs nothing but this rank's per-document lengths for
    ``seqused``/``residual``.  Both the reference core and this port therefore start from the same
    upstream call, and what the port adds is what only it reads: these frames and the operator
    outputs sized by them.  A backend that runs other kernels (the quantized attention) derives its
    own frames from the same row and builds its own entry in its own package.
    """
    cu_seqlens = row.cu_seq_q
    # ``cu_seq_k`` was folded by the backend that built the row, so it must be *this* ratio's
    # folding: a row built for another geometry would pair a boundary set with a ``seqused``
    # derived from a different one.  The tensor is one int per document, so the check is cheap.
    expected = cu_seqlens if ratio <= 1 else (cu_seqlens // ratio).to(torch.int32)
    if not torch.equal(row.cu_seq_k, expected):
        raise ValueError(
            f"the row's cu_seq_k was not folded at compress_ratio {ratio}: build the metadata "
            "through the attention backend that owns this layer's ratio, not from a row built "
            "for another geometry."
        )
    seqused = torch.diff(cu_seqlens).to(torch.int32)
    cmp_k = None
    if ratio > 0:
        cmp_k = KernelFrame(
            cu_seqlens=row.cu_seq_k,
            seqused=(seqused // ratio).to(torch.int32),
            # A ratio of one addresses the row itself: the axis is not folded, so there is no
            # trailing group and the kernels reject a value for one.
            residual=None if ratio == 1 else (seqused % ratio).to(torch.int32),
        )
    validate_compressed_frame(ratio, cmp_k)
    q = KernelFrame(cu_seqlens=cu_seqlens, seqused=seqused)
    swa_k = KernelFrame(cu_seqlens=cu_seqlens, seqused=seqused)
    return AscV41Metadata(
        cu_seq_q=cu_seqlens,
        cu_seq_k=row.cu_seq_k,
        positions_q=row.positions_q,
        exchange=row.exchange,
        ratio=ratio,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        aicpu_metadata=build_aicpu_metadata(
            ratio=ratio,
            q=q,
            swa_k=swa_k,
            cmp_k=cmp_k,
            geometry=geometry,
        ),
    )


def shard_asc_v41_metadata(
    attention_metadata: AscV41Metadata,
    *,
    cp_size: int,
    cp_rank: int,
    batch_size: int,
    seq_len: int,
) -> AscV41Metadata:
    """One rank's frames and metadata: the row's grid re-cut into this rank's chunks.

    The queries are chunk ``cp_rank`` of every document while the KV streams stay the **whole
    row** -- gathered declaratively at the AscendC attention's boundary -- so the KV frames'
    ``cu_seqlens`` address that row and their ``seqused = K_d = (cp_rank + 1) * Q_d`` is how far
    these queries reach into each document.  That is what keeps the kernels' end-aligned window
    and causal limit anchored at this rank's chunk rather than at the row's end.

    The metadata is rebuilt from the rank-local frames: the metadata is the operators'
    answer for one geometry *of one frame set*, so a rank that kept the global ones would hand
    the kernels a grid its inputs do not have.  ``seq_len`` is the **slab** width -- the shape of
    the inputs this rank holds, not of the row.
    """
    cu_seqlens = attention_metadata.cu_seq_q
    ratio = attention_metadata.ratio
    seqused = torch.diff(cu_seqlens).to(torch.int32)
    # Every document must be a whole multiple of ``cp_size * ratio``: the first factor makes
    # every prefix a whole number of chunks, the second keeps every compressed prefix a whole
    # number of groups.  The stack-wide lcm is the dataloader's business
    # (``compression_alignment``); each geometry enforces its own factor here, and together the
    # entries of a stack are the same constraint.
    # This check subsumes ``validate_compressed_frame``: a document that is a whole multiple of
    # ``cp_size * ratio`` leaves no partial group either, and the build that produced this entry
    # already ran that check at ``cp_size = 1``.
    alignment = cp_size * max(ratio, 1)
    if bool((seqused % alignment).any()):
        raise ValueError(
            f"a document length is not a multiple of cp_size * compress_ratio = {alignment}: "
            f"got {seqused.tolist()}. The dataloader's document alignment must be a multiple "
            "of that value, the same one at every context-parallel degree being compared."
        )
    chunk = (seqused // cp_size).to(torch.int32)
    # The slab is what the chunk grid has to reproduce: the row is cut into ``cp_size`` slabs and
    # the chunks are this rank's share of every document, so the two must size the same.
    if int(chunk.sum()) != batch_size * seq_len:
        raise ValueError(
            f"the sliced rows ({batch_size * seq_len}) are not the chunks' total "
            f"({int(chunk.sum())}): the boundaries describe a different row than the one the "
            "inputs were sliced from."
        )
    reach = ((cp_rank + 1) * chunk).to(torch.int32)
    exchange = cp_plan(cu_seqlens, cp_size, cp_rank, seq_len)
    q = KernelFrame(cu_seqlens=(cu_seqlens // cp_size).to(torch.int32), seqused=chunk)
    swa_k = KernelFrame(cu_seqlens=cu_seqlens, seqused=reach)
    cmp_k = None
    if ratio > 0:
        cmp_k = KernelFrame(
            cu_seqlens=attention_metadata.cu_seq_k,
            seqused=(reach // ratio).to(torch.int32),
            residual=None if ratio == 1 else (reach % ratio).to(torch.int32),
        )
    return AscV41Metadata(
        cu_seq_q=cu_seqlens,
        cu_seq_k=attention_metadata.cu_seq_k,
        positions_q=exchange.positions_q.reshape(batch_size, seq_len),
        exchange=exchange,
        ratio=ratio,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        aicpu_metadata=build_aicpu_metadata(
            ratio=ratio,
            q=q,
            swa_k=swa_k,
            cmp_k=cmp_k,
            geometry=attention_metadata.aicpu_metadata.geometry,
        ),
    )


def build_aicpu_metadata(
    *,
    ratio: int,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    geometry: AicpuGeometry,
) -> AicpuMetadata:
    """This geometry's AI_CPU operator outputs for these frames, once per microbatch.

    Only what this backend's own kernels are called with, grouped by the consumer that launches it.
    A stack that runs other kernels (the quantized attention) is another backend: it replaces this
    layer's config and builds its own entry, so nothing here has to know that it exists.
    """
    options = kernel_options(ratio=ratio, q=q, swa_k=swa_k, cmp_k=cmp_k, window_size=geometry.window_size)
    has_cmp = cmp_k is not None
    geometry_kwargs = dict(
        ori_topk=0,
        cmp_topk=geometry.index_topk if has_cmp else 0,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    sparse_attn = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        ori_topk_length=None,
        cmp_topk_length=None,
        **options,
        **geometry_kwargs,
    )
    sparse_attn_grad = torch.ops.cann_ops_transformer.sparse_flash_mla_grad_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        **options,
        **geometry_kwargs,
    )
    lightning_indexer = AscLightningIndexerMetadata(fwd=None, bwd=None)
    if has_cmp:
        # The selector's metadata belongs to the selector that replaces it, and rides this entry:
        # the teacher edge crosses both kernels, so the two share one lifetime.
        lightning_indexer = build_indexer_metadata(ratio=ratio, q=q, cmp_k=cmp_k, geometry=geometry)
    return AicpuMetadata(
        geometry=geometry,
        sparse_attn=AscSparseAttentionMetadata(fwd=sparse_attn, bwd=sparse_attn_grad),
        lightning_indexer=lightning_indexer,
    )


class _SparseMLA(torch.autograd.Function):
    """SMLAG with its teacher by-product, in the TND layout of one packed row.

    The row is the batch, so every tensor arrives with a leading 1 that the kernels do not
    take: ``squeeze(0)`` on the way in and ``reshape_as(q)`` on the way out is the whole
    translation.  A ratio-1 ``cmp_k`` is the full-resolution second stream; a ratio-2 one
    is the pooled container, wider than ``swa_k`` and addressed by ``topk_indices``.

    ``topk_scores`` is carried as an input only so the incoming gradient on it is the
    teacher edge; it is never read.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        topk_indices,
        sinks,
        attention_metadata,
        softmax_scale,
        ratio,
        window_size,
        topk_scores,
        wants_teacher,
    ):
        options = kernel_options(
            ratio=ratio,
            q=attention_metadata.q,
            swa_k=attention_metadata.swa_k,
            cmp_k=attention_metadata.cmp_k,
            window_size=window_size,
        )
        # The precomputed metadata is read instead of re-running the AI_CPU operator per layer.
        smla_metadata = attention_metadata.aicpu_metadata.sparse_attn.fwd
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            cmp_sparse_indices=topk_indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=smla_metadata,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            topk_indices,
            sinks,
            output,
            lse,
        )
        ctx.softmax_scale, ctx.ratio, ctx.window_size = softmax_scale, ratio, window_size
        ctx.wants_teacher = wants_teacher
        # The metadata is a dataclass, so ``save_for_backward`` cannot take it; it is a
        # context attribute instead.  That is not extra retention: it holds the same
        # boundary tensors the graph already keeps for the backward, and the node --
        # and with it this reference -- is released when that backward has run.
        ctx.attention_metadata = attention_metadata
        # The teacher consumes the LSE as a detached constant.
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse  # the LSE is a teacher signal, never a loss input
        # Nothing but the operand tensors is saved.  The options and the geometry are read
        # back off ``ctx.attention_metadata`` and the saved operands -- the same source the
        # forward used, which is what keeps the two kernel calls from drifting apart --
        # and the gradient metadata is built from them here.
        q, swa_k, cmp_k, topk_indices, sinks, output, lse = ctx.saved_tensors
        # ``topk_scores`` is absent on purpose: this Function takes it only so that an
        # incoming gradient on it exists, and the autograd engine routes that gradient here
        # for being an input.  Nothing is ever read off it -- the teacher comes back from
        # SMLAG below, in the carrier's own layout.
        options = kernel_options(
            ratio=ctx.ratio,
            q=ctx.attention_metadata.q,
            swa_k=ctx.attention_metadata.swa_k,
            cmp_k=ctx.attention_metadata.cmp_k,
            window_size=ctx.window_size,
        )
        # The gradient metadata rides the forward's schedule, so the backward skips its AI_CPU
        # call too.
        smla_grad_metadata = ctx.attention_metadata.aicpu_metadata.sparse_attn.bwd
        dq, dswa_k, dcmp_k, dsinks, _, cmp_softmax_l1_norm = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            ori_sparse_indices=None,
            cmp_sparse_indices=topk_indices,
            sinks=sinks,
            metadata=smla_grad_metadata,
            ori_topk_length=None,
            cmp_topk_length=None,
            softmax_scale=ctx.softmax_scale,
            **options,
        )

        # One gradient per forward input, in order: (q, swa_k, cmp_k, topk_indices,
        # sinks, attention_metadata, softmax_scale, ratio, window_size, topk_scores,
        # wants_teacher).  PyTorch silently ignores extra trailing entries, so a count
        # mismatch here would not raise -- it would quietly starve a later input, which is
        # why the list is spelled out rather than padded.
        return (
            dq,
            dswa_k,
            dcmp_k if cmp_k is not None else None,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            # Plain teacher, no loss scaling: SMLAG emits the raw marginal ``p`` and SLIKG
            # applies ``dI = Z * Y - p`` itself, so ``p`` is exactly the edge SLIKG must
            # receive.
            # Zeroing is an workaround for SMLAG kernel, SLIKG demands invalid positions be zeroed.
            cmp_softmax_l1_norm.masked_fill(topk_indices < 0, 0.0) if ctx.wants_teacher else None,
            None,
        )


class AscV41SparseAttention(CompressedSparseInnerAttention2):
    """The A5 TND kernel behind the reference forward contract.

    It is the port that carries the indexer's teacher: ``topk_scores`` is not a loss
    input but the message channel between the two kernels' backwards -- SMLAG's returns
    the raw marginal ``p`` on it, and the indexer's SLIKG backward consumes that edge to
    train the indexer.  That is why the tensor keeps threading through every layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention2.Config):
        """The fused config: the reference core's fields plus the geometry its metadata needs.

        That geometry is the *port's*: the reference core neither reads it nor knows it exists,
        and it is stated once by the override's arguments instead of being restated on every
        layer of every recipe.  The metadata built from it are the kernels' only metadata source --
        there is no per-layer fallback call -- and a recipe that states widths its layers do not
        run is a misconfiguration the kernels report, since this config cannot see them.
        """

        window_size: int
        compress_ratio: int
        # The kernel geometry has no defaults here: the released V4.1-Flash shape is stated once,
        # on the override factory a recipe names, and travels to every layer as these arguments.
        num_heads: int
        head_dim: int
        index_topk: int
        index_num_heads: int
        index_head_dim: int

        @property
        def geometry(self) -> AicpuGeometry:
            """The numbers the AI_CPU operators are called with, read off this layer's config."""
            return AicpuGeometry(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                window_size=self.window_size,
                index_topk=self.index_topk,
                index_num_heads=self.index_num_heads,
                index_head_dim=self.index_head_dim,
            )

        @property
        def attention_metadata_key(self):
            """This layer's own entry: the port's class plus its compression ratio.

            One entry per ratio is what lets the builder precompute exactly the metadata this
            geometry reads -- no stack-wide ratio table anywhere, and no entry shared by layers
            whose kernels grid different axes.
            """
            return (type(self), self.compress_ratio)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscV41Metadata:
            """This geometry's frames and AI_CPU metadata, built once per microbatch (#4897).

            The row -- and with it ``cu_seq_k``, already folded at this layer's ratio -- comes from
            the reference backend's builder, so both cores grid one set of boundaries.  The
            metadata is the AI_CPU operators' answer for that geometry, computed here at the
            eager boundary instead of per layer inside the compiled region, and rebuilt per rank
            by ``prepare_cp_metadata``.
            """
            # The parent builder is named explicitly: ``slots=True`` recreates this dataclass, so
            # a zero-argument ``super()`` would bind to the decorated-away class and raise.
            row = CompressedSparseInnerAttention2.Config.build_attention_metadata(
                self,
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            return build_fused_metadata(row, ratio=self.compress_ratio, geometry=self.geometry)

    def forward(
        self,
        q,
        swa_k,
        # Not optional, unlike the reference core's default: the kernel takes the sink
        # logits as part of its softmax denominator, and the model always has them
        # (``Attention.attn_sink`` is a parameter), so ``None`` is not a case to handle.
        attn_sink: torch.Tensor,
        attention_metadata,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
    ):
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together")
        # The teacher is a training-only by-product carried out on ``topk_scores``'
        # gradient into the indexer's SLIKG backward.  In eval there is no indexer
        # gradient to feed and the extra ``p`` would be dead output.
        wants_teacher = self.training and cmp_k is not None
        # Shapes, dtypes and the ratio's KV pairing are the kernels' own contract, checked
        # at the operator boundary rather than restated here.
        # The selection arrives document-local from the selector, which chose it with the
        # same kernel: the two share one coordinate system, and ``-1`` is how both spell an
        # unused slot.  Order is left exactly as it came -- the selector already put it in
        # position order, and the teacher's slot positions follow that order, so permuting
        # it here would move each slot's mass onto a different key.
        #
        # Layout, on the other hand, has to be translated: the model carries the selection
        # as ``[B, L, 1, K]`` while a TND kernel wants ``[T, N2, K]``.  The carrier
        # (``topk_scores``) needs no such translation -- it is already ``[T, 1, K]``, which
        # is both what SLIKG requires next to ``sparse_indices`` and what SMLA's teacher
        # gradient arrives on.
        output, _lse = _SparseMLA.apply(
            q.squeeze(0),
            swa_k.squeeze(0).unsqueeze(1),
            None if cmp_k is None else cmp_k.squeeze(0).unsqueeze(1),
            None if topk_indices is None else topk_indices.reshape(-1, 1, topk_indices.shape[-1]),
            attn_sink.float(),
            attention_metadata,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            # The carrier is translated exactly like the selection above: SLIKG needs it
            # as ``[T, 1, K]``, matching the ``sparse_indices`` it is paired with there.
            (None if (topk_scores is None or not wants_teacher) else topk_scores.reshape(-1, 1, topk_scores.shape[-1])),
            wants_teacher,
        )
        return output.reshape_as(q)
