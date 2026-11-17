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
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torchtitan.models.common.attention import (
    VarlenAttention,
    VarlenMetadata,
)

from torchtitan_npu.models.deepseek_v3_2.model import (
    SparseIndexerLoss,
    build_reference_varlen_metadata,
)
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
    cp_shard_context,
)
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss


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


class AscSparseInnerAttention(VarlenAttention):
    """Sparse attention using AscendC fused ops (MLA-absorb mode)."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
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

            The packed layout comes from the reference core's builder -- the
            fused path reads the same documents -- and the host length lists the
            TND kernels take are read here: once per microbatch for every layer
            of this geometry, instead of once per layer.
            """
            del padding_mask, max_num_documents, max_context_length
            return AscVarlenMetadata.from_varlen(build_reference_varlen_metadata(positions))

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
        if isinstance(varlen_meta, CPVarlenMetadata):
            gather = varlen_meta.k_global_gather_indices
            k_nope_T1H = k_nope_T1H.index_select(0, gather)
            k_rope_T1H = k_rope_T1H.index_select(0, gather)
            idx_k_T1H = idx_k_T1H.index_select(0, gather)

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
                attention_masks.actual_seq_qlen,
                attention_masks.actual_seq_klen,
                carrier=output_TNH,
            )

        return output_TNH.reshape(B, L_q, N, -1)


# ---------------------------------------------------------------------------
# TND varlen metadata handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AscVarlenMetadata:
    """Varlen metadata with the host offsets the TND kernels require.

    ``actual_seq_qlen``/``actual_seq_klen`` are the CANN ``actual_seq_lengths``
    form of the packed layout -- the cumulative end offsets, not per-segment
    lengths -- read off the device once per batch and kept next to the tensor
    they came from, so a rank's copy is re-read in one transfer.
    """

    varlen_meta: VarlenMetadata | CPVarlenMetadata
    actual_seq_qlen: list[int]
    actual_seq_klen: list[int]

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        """This rank's copy of the packed layout (upstream #4617's CP half).

        The global documents are re-cut for the load-balancer permutation, so the host
        length lists are re-read from the rank-local offsets: the TND kernels receive
        exactly the segments this rank runs.  The hook lives here, on the layout it
        rebuilds, rather than on the backend that produced it.

        The balancer is rebuilt here rather than received, because 0.3.0's CP entry takes
        its *type* and builds the object internally; anything but ``headtail`` means that
        entry sharded with no permutation, so this derivation passes no balancer either and
        stays in step with the shard.  An unusable type never reaches here: the pinned entry
        rejects it first (``ptrr`` included, since the map path hands it no masks).
        """
        del permutation
        context = cp_shard_context()
        if context.cp_mesh is None or context.positions is None:
            raise ValueError("prepare_cp_metadata needs the CP mesh and the pre-shard positions")
        batch_size, seq_len = context.positions.shape
        return AscVarlenMetadata.from_varlen(
            CPVarlenMetadata.from_global(
                attention_metadata.varlen_meta,
                context.cp_mesh,
                batch_size,
                seq_len,
                load_balancer=(
                    _HeadTailLoadBalancer(seq_len, context.cp_mesh.size(0), context.cp_mesh.device_type)
                    if context.load_balancer_type == "headtail"
                    else None
                ),
            )
        )

    @classmethod
    def from_varlen(cls, varlen_meta: VarlenMetadata | CPVarlenMetadata) -> AscVarlenMetadata:
        """Read both host offset lists of a packed layout in one transfer."""
        cu_q, cu_k = varlen_meta.cu_seq_q, varlen_meta.cu_seq_k
        actual_seq_qlen, actual_seq_klen = torch.stack([cu_q[1:], cu_k[1:]]).cpu().tolist()
        return cls(
            varlen_meta=varlen_meta,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
        )
