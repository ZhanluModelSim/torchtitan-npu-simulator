# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn.functional as F
from attn_gym.sparse import gather_attn
from attn_gym.sparse.indexer import lightning_indexer
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    VarlenAttention,
    VarlenMetadata,
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v3.model import (
    Attention as V3Attention,
)
from torchtitan.models.deepseek_v3.model import (
    DeepSeekV3Model,
)
from torchtitan.models.deepseek_v3.mtp import MTPDecoder
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.document_intervals import document_intervals
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

# The two forms of the packed offsets the reference core reads: the row's own documents, and
# the rank-local segments the context-parallel transform derives from them.  They are not
# related by inheritance, but they carry the same two offset tensors.
AttentionMetadata = VarlenMetadata | CPVarlenMetadata

_REFERENCE_CP_UNSUPPORTED = (
    "the DeepSeek-V3.2 reference core is not context-parallel: attn-gym's indexer "
    "selects a query's prefix of its own document, while a CP rank holds a "
    "document's middle and needs the kernels' end-aligned causal range. Use the "
    "AscendC override (override.deepseek_v3_2.sparse_attn.asc) under context "
    "parallel."
)


class ReferenceVarlenMetadata(VarlenMetadata):
    """The reference core's global packed layout, which context parallel cannot shard.

    The refusal lives here, beside the layout it refuses (upstream #4617's hook, hosted
    by the metadata rather than by the backend): a stack that cannot be sharded must say
    so while the rank-local metadata is being prepared, or the layer would silently
    attend to a global layout it does not hold.
    """

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        """Refuse context parallel while the rank-local metadata is prepared.

        attn-gym's indexer selects a query's prefix of its own document, while a CP rank
        holds a document's middle; the kernels need their end-aligned causal range, which
        this core cannot express.
        """
        del attention_metadata, permutation
        raise ValueError(_REFERENCE_CP_UNSUPPORTED)


def build_reference_varlen_metadata(positions: torch.Tensor) -> ReferenceVarlenMetadata:
    """The packed document layout both DeepSeek-V3.2 attention cores read.

    Together with :class:`ReferenceVarlenMetadata` this is the whole metadata
    contract of the reference core; the AscendC core reads the same documents
    and wraps the result, so the two backends' builders share this one call.
    """
    return ReferenceVarlenMetadata(*create_varlen_metadata_for_document(positions))


@functools.cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    assert dim & (dim - 1) == 0, "Hadamard dim must be a power of two"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        index_n_heads: int
        index_head_dim: int
        rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: RoPE.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk

        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        qr_BLD: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x_BLD.size()

        q_BLNH = self.wq_b(qr_BLD)
        with spmd.local():
            q_BLNH = q_BLNH.view(bsz, seqlen, -1, self.head_dim)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    q_BLNH,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )
        q_pe_BLNH, q_nope_BLNH = torch.split(q_BLNH, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_BLD = self.k_norm(self.wk(x_BLD))
        k_pe_BLD, k_nope_BLD = torch.split(k_BLD, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe_BLNH, k_pe_BL1H = self.rope(q_pe_BLNH, k_pe_BLD.unsqueeze(2), positions)
        idx_q_BLNH = Indexer._hadamard_rotate(torch.cat([q_pe_BLNH, q_nope_BLNH], dim=-1))
        idx_k_BL1H = Indexer._hadamard_rotate(torch.cat([k_pe_BL1H.squeeze(2), k_nope_BLD], dim=-1))

        idx_w_BLN = self.weights_proj(x_BLD) * (self.n_heads**-0.5)
        idx_w_BLN = idx_w_BLN * (self.head_dim**-0.5)

        return idx_q_BLNH, idx_k_BL1H, idx_w_BLN

    @staticmethod
    def select(
        idx_q_BLNH: torch.Tensor,
        idx_k_BL1H: torch.Tensor,
        idx_w_BLN: torch.Tensor,
        attention_masks: AttentionMetadata,
        index_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The top-k selection and the student's logits at the slots it picked.

        Returns ``(topk_indices_BLqK, topk_scores_BLqK)``:

        * the indices are numbered inside the query's own document, ``-1`` for a slot the
          selection left unused -- the numbering ``gather_attn`` reads back, and the fused
          kernels emit.  Causality and document isolation are the indexer's own
          ``causal``/``cu_seqlens`` contract, so no mask is materialized here;
        * the scores are the indexer's own logits ``sum_h w * relu(q . k)`` at those slots.
          Selection is discrete and ``lightning_indexer`` returns indices alone, so this
          second pass -- over ``K`` slots rather than the dense ``[B, Lq, Lkv]`` grid -- is
          the one that carries the indexer's gradient.

        The pass that picks runs under ``no_grad``; the pass that scores does not.
        """
        batch, seqlen, num_heads, head_dim = idx_q_BLNH.shape
        tokens = batch * seqlen
        # ``idx_w`` carries ``(H * D) ** -0.5`` because the fused NPU indexer reads the weights
        # that way, while attn-gym's indexer applies that factor to its own score; it is taken
        # back out for this one call, or the factor would be applied twice.  The scores below
        # keep the folded convention, which is the one the fused kernel uses as well.
        #
        # attn-gym validates this call's weights against the queries' dtype, so it keeps the
        # module's (its reference indexer upcasts its own accumulation to fp32); the score pass
        # below is spelled in fp32 the way this model's removed dense scores were.
        selection_weights_BLN = idx_w_BLN * math.sqrt(idx_w_BLN.size(-1) * idx_k_BL1H.size(-1))
        with torch.no_grad():
            topk_indices_BLqK = lightning_indexer(
                idx_q_BLNH.reshape(1, tokens, num_heads, head_dim),
                idx_k_BL1H.reshape(1, tokens, -1),
                selection_weights_BLN.reshape(1, tokens, num_heads),
                index_topk,
                causal=True,
                compress_ratio=1,
                cu_seqlens=attention_masks.cu_seq_q,
                cu_seqlens_k=attention_masks.cu_seq_k,
                impl="reference",
            ).reshape(batch, seqlen, -1)

        # The picks are numbered inside their own document while the keys are stored in the
        # packed row, so the shift between the two numberings is the document's pool start.
        # A ``-1`` slot may name a real entry once shifted; its score is masked below.
        pool_start_T1 = document_intervals(
            attention_masks.cu_seq_q,
            attention_masks.cu_seq_k,
            num_tokens=tokens,
        )[0]
        slots_TK = topk_indices_BLqK.reshape(tokens, -1).long() + pool_start_T1
        picked_TKD = idx_k_BL1H.reshape(tokens, -1).float()[slots_TK.clamp_min(0)]
        dots_THK = torch.einsum("thd,tkd->thk", idx_q_BLNH.reshape(tokens, num_heads, head_dim).float(), picked_TKD)
        scores_TK = (dots_THK.relu() * idx_w_BLN.reshape(tokens, num_heads).float().unsqueeze(-1)).sum(dim=1)
        return topk_indices_BLqK, scores_TK.reshape(batch, seqlen, -1).masked_fill(topk_indices_BLqK < 0, -torch.inf)

    @staticmethod
    def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d, device=x.device, dtype=x.dtype)
        return F.linear(x, H) * (d**-0.5)


# The reference indexer loss guards two different spaces, and they cannot share one constant:
#
# * ``_MASKED_LOGIT`` lives in *logit* space: it stands in for ``-inf`` on a slot the selection left
#   unused, so both softmaxes stay finite and the masked slot gets exactly zero mass instead of
#   producing ``0 * -inf = NaN`` (which ``torch.where`` would carry through the unselected branch's
#   backward).  It has to stay far below every real logit at all times, which is why it is enormous
#   rather than the logarithm of some small probability.
# * ``_MIN_TEACHER_MASS`` lives in *probability* space: the masked slot's mass is exactly zero, and
#   the teacher's own ``log`` would be ``-inf`` there.  Its weight is zero too, so the value only
#   has to be finite.
#
# Deriving one from the other would tie the mask's depth to the log guard (``log(1e-30) == -69`` is
# not below a logit of, say, ``-100``), so they stay separate.
_MASKED_LOGIT = -1e30
_MIN_TEACHER_MASS = 1e-30


class SparseIndexerLoss(AuxLoss):
    """Indexer alignment loss with gradient injection.

    Both distributions live on the selected slots alone: the caller hands over the
    attention's logits and the indexer's logits at the top-k positions, and no dense
    ``[B, Lq, Lkv]`` tensor is built on either side.

    The selection arrives exactly as :meth:`Indexer.select` emits it -- numbered inside each
    query's own document, ``-1`` for a slot it left unused -- together with the metadata whose
    offsets say where those documents sit in the packed row, because the teacher reads the
    keys' **rows** while the selection is numbered per document; the shift is applied here,
    where the two numberings meet.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        coeff: float = 1.0
        # The indexer loss is additive over tokens, so ranks contribute distinct
        # data: reduce over the loss mesh (dp+cp), not only the batch mesh.
        reduce_mesh: str = "loss"

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BL1H: torch.Tensor,
        scale: float,
        topk_indices_BLqK: torch.Tensor,
        topk_scores_BLqK: torch.Tensor,
        attention_masks: AttentionMetadata,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        # A slot the selection left unused is masked out of both distributions with a large finite
        # negative rather than ``-inf``: ``0 * -inf`` is NaN, and ``torch.where`` would carry that
        # NaN through the unselected branch's backward.
        valid_BLqK = topk_indices_BLqK >= 0
        # The selection is numbered inside its own document; the teacher reads the keys of the
        # packed row, so the shift between the two numberings is applied here, where the
        # selection arrives.
        pool_start_BL1 = document_intervals(
            attention_masks.cu_seq_q,
            attention_masks.cu_seq_k,
            num_tokens=topk_indices_BLqK.size(0) * topk_indices_BLqK.size(1),
        )[0].reshape((*topk_indices_BLqK.shape[:2], 1))
        selected_rows_BLqK = torch.where(
            valid_BLqK,
            topk_indices_BLqK + pool_start_BL1,
            topk_indices_BLqK,
        )
        slots_BLqK = selected_rows_BLqK.clamp(min=0)

        # Teacher: the attention's own distribution over the selected slots, taken as
        # log-probabilities of the same logits -- not ``log(softmax(...))``, which underflows to
        # ``-inf`` exactly where the student is peaked.  It is constant with respect to the indexer:
        # the caller hands over detached q/k and the gradient arrives through ``carrier``.  This
        # model has no window and no sink, so the selection is the whole support, every head's mass
        # is one, and the teacher is the head-averaged softmax.
        # The keys are read as one packed row -- the ``B * L`` tokens the metadata's offsets
        # number -- rather than per batch element, so a selected row is addressed the same way
        # whichever document and whichever batch element it came from.
        k_sqz = k_BL1H.float().squeeze(2).reshape(-1, k_BL1H.size(-1))
        k_gathered = k_sqz[slots_BLqK]
        teacher_logits = torch.einsum("blhd,blkd->blhk", q_BLNH.float(), k_gathered).mul(scale)
        teacher_logits = teacher_logits.masked_fill(~valid_BLqK.unsqueeze(2), _MASKED_LOGIT)
        teacher = F.log_softmax(teacher_logits, dim=-1).exp().mean(dim=2)

        # Student: the indexer's log-probabilities over the same slots under the same mask, so its
        # softmax normalizes exactly where the operator's does.  No epsilon or clamp appears on this
        # path, so autograd yields the operator's ``Y - teacher`` exactly, saturated slots included.
        # The selection marks an unused slot ``-inf``; the finite sentinel replaces it, so a row
        # whose every slot is unused still gets a finite softmax instead of a NaN one.
        student_logits = topk_scores_BLqK.float().masked_fill(~valid_BLqK, _MASKED_LOGIT)
        log_student = F.log_softmax(student_logits, dim=-1)

        # ``teacher * (log teacher - log student)`` is the teacher-weighted KL, summed over this
        # microbatch's query rows.  The clamp only guards a mass that fp32 could underflow to zero;
        # a row whose every slot is unused is uniform on both sides and contributes exactly zero.
        kl = (teacher * (teacher.clamp_min(_MIN_TEACHER_MASS).log() - log_student)).sum(dim=-1)
        # ``AuxLoss`` normalizes by the step's global valid-token count, the same denominator the
        # main loss uses, so what is handed over is that sum rather than a mean already divided by
        # the row count.
        return self.inject(kl.sum(), carrier=carrier)


class SparseInnerAttention(VarlenAttention):
    """Sparse attention core for DSA, with separated nope/rope support.

    The core is two attn-gym calls: :func:`lightning_indexer` picks the slots and
    :func:`gather_attn` attends to exactly them, so the selection *is* the softmax's whole
    support -- this model has no sliding window and no sink.  The value is the KV latent's
    un-rotated prefix: MLA's absorbed value is ``k_nope``, and attn-gym carries one tensor
    per KV source, so the same ``[nope | rope]`` latent is the key and (for its ``nope``
    prefix) the value; the rope tail of the attention output is dropped.

    Both calls read the packed layout the varlen metadata describes, one document per
    segment, and both number the pool the same way -- the indexer emits document-local
    indices, at ratio 1 the row is its own candidate pool, and ``gather_attn`` reads them
    back in that numbering.  Only the unsharded layout is supported: attn-gym's causal
    selection is a query's *prefix* of its document, while a context-parallel rank holds a
    document's middle, where the fused kernels' end-aligned convention is required.

    This is the reference core: attn-gym's eager implementations do not chunk, so the
    indexer materializes its ``[B, T, H, T]`` score and ``gather_attn`` its
    ``[B, kv_heads, T, topk, D]`` slots in fp32, and the footprint grows with ``T * topk``
    (one 2048/2048/128-head layer measured 26.75 GiB on 910C).  Training runs the fused
    AscendC kernels instead -- ``override.deepseek_v3_2.sparse_attn.asc``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        index_topk: int
        indexer_loss: SparseIndexerLoss.Config = field(default_factory=SparseIndexerLoss.Config)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> ReferenceVarlenMetadata:
            """The packed document layout the reference core reads.

            Upstream #4897 makes the attention backend own this construction, so
            the decoder builds it once per microbatch and every layer of this
            geometry shares it.
            """
            del padding_mask, max_num_documents, max_context_length
            return build_reference_varlen_metadata(positions)

    def __init__(self, config: Config):
        super().__init__(config)
        self.index_topk = config.index_topk
        self.indexer_loss = config.indexer_loss.build()

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q_nope_BLNH: torch.Tensor,
        k_nope_BL1H: torch.Tensor,
        q_rope_BLNH: torch.Tensor,
        k_rope_BL1H: torch.Tensor,
        idx_q_BLNH: torch.Tensor,
        idx_k_BL1H: torch.Tensor,
        idx_w_BLN: torch.Tensor,
        *,
        attention_masks: AttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        if isinstance(attention_masks, CPVarlenMetadata):
            raise ValueError(_REFERENCE_CP_UNSUPPORTED)
        q_BLNH = torch.cat([q_nope_BLNH, q_rope_BLNH], dim=-1)
        k_BL1H = torch.cat([k_nope_BL1H, k_rope_BL1H], dim=-1)

        topk_indices_BLqK, topk_scores_BLqK = Indexer.select(
            idx_q_BLNH,
            idx_k_BL1H,
            idx_w_BLN,
            attention_masks,
            self.index_topk,
        )

        output_BLNH = self._sparse_attention(
            q_BLNH,
            k_BL1H,
            topk_indices_BLqK,
            attention_masks,
            k_nope_BL1H.size(-1),
            scale,
        )
        if self.training:
            output_BLNH = self.indexer_loss(
                q_BLNH.detach(),
                k_BL1H.detach(),
                scale,
                topk_indices_BLqK,
                topk_scores_BLqK,
                attention_masks,
                carrier=output_BLNH,
            )
        return output_BLNH

    @staticmethod
    def _sparse_attention(
        q_BLNH: torch.Tensor,
        k_BL1H: torch.Tensor,
        topk_indices_BLqK: torch.Tensor,
        attention_masks: AttentionMetadata,
        value_head_dim: int,
        scale: float,
    ) -> torch.Tensor:
        """One masked softmax over the selected slots, as a single ``gather_attn`` call.

        The batch is packed into one row, which is the numbering the metadata's offsets
        describe.  ``sliding_window_size=0`` leaves the local branch with no slot of its
        own, so the sparse pool is the entire support; the same latent is passed for both
        KV sources because a ``[nope | rope]`` key is also its own value on the ``nope``
        prefix.  ``local_kv`` still has to be a query-length tensor for attn-gym's shape
        check, which the key itself already is.
        """
        batch, seqlen, _, head_dim = q_BLNH.shape
        tokens = batch * seqlen
        kv_B1TD = k_BL1H.reshape(1, tokens, 1, head_dim).transpose(1, 2)
        output_B1TD = gather_attn(
            q_BLNH.reshape(1, tokens, -1, head_dim).transpose(1, 2),
            kv_B1TD,
            kv_B1TD,
            topk_indices_BLqK.reshape(1, tokens, -1),
            None,
            sliding_window_size=0,
            cu_seqlens=attention_masks.cu_seq_q,
            cu_seqlens_k=attention_masks.cu_seq_k,
            scale=scale,
            impl="reference",
        )
        assert isinstance(output_B1TD, torch.Tensor)
        return output_B1TD.transpose(1, 2).reshape(batch, seqlen, -1, head_dim)[..., :value_head_dim]


class Attention(V3Attention):
    """Multi-head latent attention in MQA absorb mode for DeepSeek-V3.2."""

    @dataclass(kw_only=True, slots=True)
    class Config(V3Attention.Config):
        w_uk: BatchedLinear.Config
        w_uv: BatchedLinear.Config
        indexer: Indexer.Config

    def __init__(self, config: Config):
        super().__init__(config)
        del self.wkv_b

        self.w_uk = config.w_uk.build()
        self.w_uv = config.w_uv.build()
        self.indexer = config.indexer.build()

        self.register_state_dict_post_hook(self._merge_wkv_b_on_save)
        self.register_load_state_dict_pre_hook(self._split_wkv_b_on_load)

    @staticmethod
    def _redistribute_state_dict_tensor(
        tensor: torch.Tensor,
        placements=None,
    ) -> torch.Tensor:
        """Redistribute a state-dict tensor to the requested placements."""
        if not isinstance(tensor, DTensor):
            return tensor
        if placements is None:
            placements = [Replicate()] * tensor.device_mesh.ndim
        return tensor.redistribute(tensor.device_mesh, placements)

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x_BLD.size()

        qr_BLD = self.q_norm(self.wq_a(x_BLD))
        q_BLNH = self.wq_b(qr_BLD)
        with spmd.local():
            q_BLNH = q_BLNH.view(bsz, seqlen, -1, self.qk_head_dim)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    q_BLNH,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )

        q_nope_BLNH, q_pe_BLNH = torch.split(q_BLNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_BLD = self.wkv_a(x_BLD)
        kv_nope_BL1H, k_pe_BL1H = torch.split(kv_BLD.unsqueeze(2), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope_BL1H = self.kv_norm(kv_nope_BL1H)

        q_pe_BLNH, k_pe_BL1H = self.rope(q_pe_BLNH, k_pe_BL1H, positions)
        q_nope_BLNH = self.w_uk(q_nope_BLNH)

        idx_q_BLNH, idx_k_BL1H, idx_w_BLN = self.indexer(x_BLD.detach(), qr_BLD.detach(), positions=positions)
        # ``spmd_types`` cannot yet express the required TP-CP redistribution
        # for ``idx_k``, so mark the TP transition explicitly.
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            spmd.mutate_type(idx_k_BL1H, "tp", src=spmd.R, dst=spmd.I)

        output_BLNH = self.inner_attention(
            q_nope_BLNH,
            kv_nope_BL1H,
            q_pe_BLNH,
            k_pe_BL1H,
            idx_q_BLNH,
            idx_k_BL1H,
            idx_w_BLN,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
        )

        output_BLNH = self.w_uv(output_BLNH)
        output_BLNH = output_BLNH.contiguous().view(bsz, seqlen, -1)
        return self.wo(output_BLNH)

    @staticmethod
    def _split_wkv_b_on_load(module, state_dict, prefix, *args):
        wkv_key = f"{prefix}wkv_b.weight"
        wkv_b = state_dict.pop(wkv_key)
        placements = wkv_b.placements if isinstance(wkv_b, DTensor) else None
        wkv_b = Attention._redistribute_state_dict_tensor(wkv_b)

        wkv_b_3d = wkv_b.view(module.n_heads, -1, module.kv_lora_rank)
        w_uk = wkv_b_3d[:, : module.qk_nope_head_dim, :].transpose(-2, -1).contiguous()
        w_uv = wkv_b_3d[:, module.qk_nope_head_dim :, :].contiguous()
        w_uk = w_uk.reshape(-1, module.qk_nope_head_dim)
        w_uv = w_uv.reshape(-1, module.kv_lora_rank)

        state_dict[f"{prefix}w_uk.weight"] = Attention._redistribute_state_dict_tensor(w_uk, placements)
        state_dict[f"{prefix}w_uv.weight"] = Attention._redistribute_state_dict_tensor(w_uv, placements)

    @staticmethod
    def _merge_wkv_b_on_save(module, state_dict, prefix, local_metadata):
        w_uk = state_dict.pop(f"{prefix}w_uk.weight")
        w_uv = state_dict.pop(f"{prefix}w_uv.weight")
        placements = w_uk.placements if isinstance(w_uk, DTensor) else None
        w_uk = Attention._redistribute_state_dict_tensor(w_uk)
        w_uv = Attention._redistribute_state_dict_tensor(w_uv)

        w_uk = w_uk.view(module.n_heads, module.kv_lora_rank, module.qk_nope_head_dim)
        w_uv = w_uv.view(module.n_heads, module.v_head_dim, module.kv_lora_rank)
        wkv_b = torch.cat([w_uk.transpose(-2, -1), w_uv], dim=1).reshape(-1, module.kv_lora_rank)
        wkv_b = Attention._redistribute_state_dict_tensor(wkv_b, placements)
        state_dict[f"{prefix}wkv_b.weight"] = wkv_b.contiguous()


class DeepSeekV32Model(DeepSeekV3Model):
    """DeepSeek-V3.2 model based on DeepSeek V3.

    The model replaces MLA with DSA and applies V3.2-specific sharding.

    The model carries no metadata seam of its own: its attention backends build
    the per-microbatch metadata (``SparseInnerAttention.Config``, or the
    AscendC override's ``AscSparseInnerAttention.Config``), and the decoder
    builds and dispatches it as upstream #4897 does.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        def update_from_config(self, *, config, **kwargs):
            if hasattr(config, "training"):
                seq_len = config.training.seq_len
                for _, rope_cfg, _, _ in self.traverse(RoPE.Config):
                    setattr(rope_cfg, "max_seq_len", seq_len)  # noqa: B010

            MTPDecoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan_npu.models.deepseek_v3_2.sharding import (
                set_deepseek_v3_2_sharding_config,
            )

            set_deepseek_v3_2_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )
