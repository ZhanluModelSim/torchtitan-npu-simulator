# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V3.2 sparse attention with fused AscendC kernels.

The TND metadata the fused kernels take and the fused attention core live
together here: both belong to the same backend, so the backend's config builds
the metadata (upstream #4897) and its forward consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_npu
from torchtitan.models.common.attention import (
    VarlenInnerAttention,
    VarlenMetadata,
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.aux_loss import AuxLoss

from torchtitan_npu.models.deepseek_v3_2.model import SparseIndexerLoss


@spmd.register_autograd_function
class _AscSparseIndexerLossFunc(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q_nope_TNH,
        k_nope_T1H,
        idx_q_TNH,
        idx_k_T1H,
        idx_w_TN,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        q_rope_TNH,
        k_rope_T1H,
        actual_seq_qlen,
        actual_seq_klen,
        acc_buffer,
    ):
        ctx.save_for_backward(
            q_nope_TNH,
            k_nope_T1H,
            idx_q_TNH,
            idx_k_T1H,
            idx_w_TN,
            sparse_indices,
            softmax_max,
            softmax_sum,
            q_rope_TNH,
            k_rope_T1H,
        )
        ctx.scale_value = scale_value
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_klen = actual_seq_klen
        ctx.acc_buffer = acc_buffer
        return torch.zeros((), device=q_nope_TNH.device, dtype=torch.float32)

    @staticmethod
    def typecheck_forward(
        q_nope_TNH,
        k_nope_T1H,
        idx_q_TNH,
        idx_k_T1H,
        idx_w_TN,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        q_rope_TNH,
        k_rope_T1H,
        actual_seq_qlen,
        actual_seq_klen,
        acc_buffer,
    ):
        return _AscSparseIndexerLossFunc.apply(
            q_nope_TNH,
            k_nope_T1H,
            idx_q_TNH,
            idx_k_T1H,
            idx_w_TN,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value,
            q_rope_TNH,
            k_rope_T1H,
            actual_seq_qlen,
            actual_seq_klen,
            acc_buffer,
        )

    @staticmethod
    def backward(ctx, grad_scale):  # pyrefly: ignore [bad-override]
        (
            q_nope_TNH,
            k_nope_T1H,
            idx_q_TNH,
            idx_k_T1H,
            idx_w_TN,
            sparse_indices,
            softmax_max,
            softmax_sum,
            q_rope_TNH,
            k_rope_T1H,
        ) = ctx.saved_tensors

        d_q_idx, d_k_idx, d_w, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            q_nope_TNH,
            k_nope_T1H,
            idx_q_TNH,
            idx_k_T1H,
            idx_w_TN,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value=ctx.scale_value,
            query_rope=q_rope_TNH,
            key_rope=k_rope_T1H,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_klen=ctx.actual_seq_klen,
            layout="TND",
            sparse_mode=3,
        )

        # ``grad_scale`` arrives from ``AuxLoss.inject`` and already carries
        # ``coeff / global_valid_tokens`` -- one factor for the coefficient and one
        # for the step's token count, the denominator the main loss uses.
        d_q_idx = d_q_idx * grad_scale
        d_k_idx = d_k_idx * grad_scale
        d_w = d_w * grad_scale

        # The kernel answers with the summed KL only on the backward, so the metric
        # is accumulated here; ``inject``'s forward accumulation adds the zero
        # placeholder it was handed.
        denominator = AuxLoss._step_denominator
        assert denominator is not None, "AuxLoss.set_step_denominator() must be called"
        ctx.acc_buffer.add_(loss.squeeze() / denominator)

        return (
            None,
            None,
            d_q_idx,
            d_k_idx,
            d_w,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AscSparseIndexerLoss(SparseIndexerLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(SparseIndexerLoss.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q_nope_TNH,
        k_nope_T1H,
        idx_q_TNH,
        idx_k_T1H,
        idx_w_TN,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale,
        q_rope_TNH,
        k_rope_T1H,
        actual_seq_qlen,
        actual_seq_klen,
        *,
        carrier,
    ):
        dummy = _AscSparseIndexerLossFunc.apply(
            q_nope_TNH,
            k_nope_T1H,
            idx_q_TNH,
            idx_k_T1H,
            idx_w_TN,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale,
            q_rope_TNH,
            k_rope_T1H,
            actual_seq_qlen,
            actual_seq_klen,
            self.instance_acc,
        )
        # ``AuxLoss`` owns the per-step normalization: the summed KL goes in raw,
        # and the coefficient and the global valid-token count are applied inside.
        return self.inject(dummy, carrier=carrier)


class AscSparseInnerAttention(VarlenInnerAttention):
    """Sparse attention using AscendC fused ops (MLA-absorb mode)."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenInnerAttention.Config):
        index_topk: int
        indexer_loss: AscSparseIndexerLoss.Config

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscVarlenMetadata:
            """The TND metadata this backend consumes (upstream #4897).

            The packed layout comes from the same upstream factory the
            reference core uses -- the fused path reads the same documents --
            and the host length list the TND kernels take is read here: once
            per microbatch for every layer of this geometry, instead of once
            per layer.
            """
            del padding_mask, max_num_documents, max_context_length
            return AscVarlenMetadata.from_varlen(create_varlen_metadata_for_document(positions))

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.index_topk = config.index_topk
        self.indexer_loss = AscSparseIndexerLoss(config.indexer_loss)

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
        attention_masks: AscVarlenMetadata,
        scale: float,
    ) -> torch.Tensor:
        B, L_q, N, _ = q_nope_BLNH.shape
        _, L_k, _, _ = k_nope_BL1H.shape
        _, _, N2, _ = idx_q_BLNH.shape
        T_q = B * L_q
        T_k = B * L_k

        q_nope_TNH = q_nope_BLNH.reshape(T_q, N, -1)
        k_nope_T1H = k_nope_BL1H.reshape(T_k, 1, -1)
        q_rope_TNH = q_rope_BLNH.reshape(T_q, N, -1)
        k_rope_T1H = k_rope_BL1H.reshape(T_k, 1, -1)
        idx_q_TNH = idx_q_BLNH.reshape(T_q, N2, -1)
        idx_k_T1H = idx_k_BL1H.reshape(T_k, 1, -1)
        idx_w_TN = idx_w_BLN.reshape(T_q, N2)

        varlen_meta = attention_masks.varlen_meta
        cu_seq_q_1 = varlen_meta.cu_seq_q[1:]
        cu_seq_k_1 = varlen_meta.cu_seq_k[1:]

        sparse_indices, _ = torch_npu.npu_lightning_indexer(
            idx_q_TNH,
            idx_k_T1H,
            idx_w_TN,
            actual_seq_lengths_query=cu_seq_q_1,
            actual_seq_lengths_key=cu_seq_k_1,
            layout_query="TND",
            layout_key="TND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            return_value=False,
        )

        output_TNH, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
            q_nope_TNH,
            k_nope_T1H,
            k_nope_T1H,
            sparse_indices,
            scale,
            actual_seq_lengths_query=cu_seq_q_1,
            actual_seq_lengths_kv=cu_seq_k_1,
            query_rope=q_rope_TNH,
            key_rope=k_rope_T1H,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )

        if self.training:
            output_TNH = self.indexer_loss(
                q_nope_TNH,
                k_nope_T1H,
                idx_q_TNH,
                idx_k_T1H,
                idx_w_TN,
                sparse_indices,
                softmax_max,
                softmax_sum,
                scale,
                q_rope_TNH,
                k_rope_T1H,
                attention_masks.actual_seq_lengths,
                attention_masks.actual_seq_lengths,
                carrier=output_TNH,
            )

        return output_TNH.reshape(B, L_q, N, -1)


# ---------------------------------------------------------------------------
# TND varlen metadata handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AscVarlenMetadata:
    """Varlen metadata with the host offsets the TND kernels require.

    ``actual_seq_lengths`` is the CANN ``actual_seq_lengths`` form of the packed
    layout -- the cumulative end offsets, not per-segment lengths -- read off the
    device once per batch and kept next to the tensor it came from, so the host
    list is read in one transfer.  One list serves both the query and the key
    kernel argument: ``create_varlen_metadata_for_document`` hands the same
    packed offsets to ``cu_seq_q`` and ``cu_seq_k``.
    """

    varlen_meta: VarlenMetadata
    actual_seq_lengths: list[int]

    @classmethod
    def from_varlen(cls, varlen_meta: VarlenMetadata) -> AscVarlenMetadata:
        """Read the host offset list of a packed layout in one transfer."""
        # ``cu_seq_q`` is read alone because the pinned upstream factory assigns
        # the identical ``packed_cu_seqlens`` tensor to ``cu_seq_q`` and ``cu_seq_k``.
        actual_seq_lengths = varlen_meta.cu_seq_q[1:].cpu().tolist()
        return cls(
            varlen_meta=varlen_meta,
            actual_seq_lengths=actual_seq_lengths,
        )
