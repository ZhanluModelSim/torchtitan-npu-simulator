from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_npu

from torchtitan.config import derive, override
from torchtitan.models.common.attention import (
    AttentionMasksType,
    VarlenAttention,
    VarlenMetadata,
)

from torchtitan_npu_lite.models.deepseek_v3_2.model import SparseIndexerLoss, SparseInnerAttention
from torchtitan_npu_lite.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata
from torchtitan_npu_lite.patches.torchtitan.models.common.mask_handler import BaseMaskHandler


@dataclass(frozen=True, eq=False)
class NPUVarlenMetadata:
    """Carries per-rank VarlenMetadata + pre-extracted CPU int lists for NPU kernels.

    ``npu_sparse_lightning_indexer_grad_kl_loss`` requires ``actual_seq_qlen``
    and ``actual_seq_klen`` as CPU ``list[int]`` in TND mode, while the source
    ``VarlenMetadata.cu_seq_q/k`` are device tensors. This wrapper extracts them
    via a single fused D2H transfer in ``NPUVarlenMetadataHandler.post_process``
    to minimize host-device synchronization.
    """

    varlen_meta: VarlenMetadata | CPVarlenMetadata
    actual_seq_qlen: list[int]
    actual_seq_klen: list[int]


class NPUVarlenMetadataHandler(BaseMaskHandler):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseMaskHandler.Config):
        pass

    def post_process(self, masks: AttentionMasksType) -> NPUVarlenMetadata:
        assert isinstance(masks, (VarlenMetadata, CPVarlenMetadata)), (
            f"expected VarlenMetadata or CPVarlenMetadata, got {type(masks)}"
        )
        cu_q, cu_k = masks.cu_seq_q, masks.cu_seq_k
        actual_seq_qlen, actual_seq_klen = (
            torch.stack([cu_q[1:], cu_k[1:]]).cpu().tolist()
        )
        return NPUVarlenMetadata(
            varlen_meta=masks,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
        )


@spmd.register_autograd_function
class _NPUSparseIndexerLossFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
                sparse_indices, softmax_max, softmax_sum,
                scale_value, q_rope_TNH, k_rope_T1H,
                actual_seq_qlen, actual_seq_klen,
                coeff, acc_buffer):
        ctx.save_for_backward(q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
                              sparse_indices, softmax_max, softmax_sum,
                              q_rope_TNH, k_rope_T1H)
        ctx.scale_value = scale_value
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_klen = actual_seq_klen
        ctx.coeff = coeff
        ctx.acc_buffer = acc_buffer
        return torch.zeros((), device=q_nope_TNH.device, dtype=torch.float32)

    @staticmethod
    def typecheck_forward(
        q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
        sparse_indices, softmax_max, softmax_sum,
        scale_value, q_rope_TNH, k_rope_T1H,
        actual_seq_qlen, actual_seq_klen,
        coeff, acc_buffer,
    ):
        return _NPUSparseIndexerLossFunc.apply(
            q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
            sparse_indices, softmax_max, softmax_sum,
            scale_value, q_rope_TNH, k_rope_T1H,
            actual_seq_qlen, actual_seq_klen,
            coeff, acc_buffer,
        )

    @staticmethod
    def backward(ctx, grad_scale):
        (q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
         sparse_indices, softmax_max, softmax_sum,
         q_rope_TNH, k_rope_T1H) = ctx.saved_tensors

        d_q_idx, d_k_idx, d_w, loss = (
            torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
                sparse_indices, softmax_max, softmax_sum,
                scale_value=ctx.scale_value,
                query_rope=q_rope_TNH, key_rope=k_rope_T1H,
                actual_seq_qlen=ctx.actual_seq_qlen,
                actual_seq_klen=ctx.actual_seq_klen,
                layout="TND", sparse_mode=3,
            )
        )

        # NPU kernel returns sum_KL gradients and loss, scale to get mean_KL.
        # grad_scale = coeff / (n_tokens * gbs) from inject's divide-by-seqlen.
        d_q_idx = d_q_idx * grad_scale
        d_k_idx = d_k_idx * grad_scale
        d_w = d_w * grad_scale

        ctx.acc_buffer.add_(loss.squeeze() * grad_scale / ctx.coeff)

        return (None, None, d_q_idx, d_k_idx, d_w, None, None, None,
                None, None, None, None, None, None, None)


class NPUSparseIndexerLoss(SparseIndexerLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(SparseIndexerLoss.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def forward(self, q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
                sparse_indices, softmax_max, softmax_sum,
                scale, q_rope_TNH, k_rope_T1H,
                actual_seq_qlen, actual_seq_klen, *, carrier):
        dummy = _NPUSparseIndexerLossFunc.apply(
            q_nope_TNH, k_nope_T1H, idx_q_TNH, idx_k_T1H, idx_w_TN,
            sparse_indices, softmax_max, softmax_sum,
            scale, q_rope_TNH, k_rope_T1H,
            actual_seq_qlen, actual_seq_klen,
            self.coeff, self._acc,
        )
        # NPU kernel returns sum_KL while logger expects mean_KL.
        return self.inject(carrier, dummy / q_nope_TNH.shape[0])


class NPUSparseInnerAttention(VarlenAttention):
    """Sparse attention using NPU fused ops (MLA-absorb mode)."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        index_topk: int
        indexer_loss: NPUSparseIndexerLoss.Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.index_topk = config.index_topk
        self.indexer_loss = NPUSparseIndexerLoss(config.indexer_loss)

    def forward(
        self,
        q_nope_BLNH: torch.Tensor,
        k_nope_BL1H: torch.Tensor,
        q_rope_BLNH: torch.Tensor,
        k_rope_BL1H: torch.Tensor,
        idx_q_BLNH: torch.Tensor,
        idx_k_BL1H: torch.Tensor,
        idx_w_BLN: torch.Tensor,
        *,
        attention_masks: NPUVarlenMetadata,
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
                q_nope_TNH, k_nope_T1H,
                idx_q_TNH, idx_k_T1H, idx_w_TN,
                sparse_indices, softmax_max, softmax_sum,
                scale, q_rope_TNH, k_rope_T1H,
                attention_masks.actual_seq_qlen,
                attention_masks.actual_seq_klen,
                carrier=output_TNH,
            )

        return output_TNH.reshape(B, L_q, N, -1)


@override(
    target=SparseInnerAttention.Config,
    description="NPU sparse flash attention via torch_npu.npu_sparse_flash_attention",
)
def kernel(
    cfg: SparseInnerAttention.Config,
) -> NPUSparseInnerAttention.Config:
    return NPUSparseInnerAttention.Config(
        index_topk=cfg.index_topk,
        indexer_loss=derive(cfg.indexer_loss, NPUSparseIndexerLoss.Config),
    )


@override(
    target=BaseMaskHandler.Config,
    description="NPUVarlenMetadata handler for TND layout (NPU sparse attention).",
)
def mask_handler(
    cfg: BaseMaskHandler.Config,
) -> NPUVarlenMetadataHandler.Config:
    return NPUVarlenMetadataHandler.Config()
