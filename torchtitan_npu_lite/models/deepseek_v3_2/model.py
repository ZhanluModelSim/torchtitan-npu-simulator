# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_mask
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.attention import AttentionMasksType, FlexAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v3.model import (
    Attention as V3Attention,
    DeepSeekV3Model,
)
from torchtitan.protocols.module import Module

from torchtitan_npu_lite.patches.torchtitan.models.common.aux_loss import LoggedAuxLoss
from torchtitan_npu_lite.patches.torchtitan.models.common.linear import BatchedLinear
from torchtitan_npu_lite.patches.torchtitan.models.common.mask_handler import (
    BaseMaskHandler,
)


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
        q_pe_BLNH, q_nope_BLNH = torch.split(
            q_BLNH, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_BLD = self.k_norm(self.wk(x_BLD))
        k_pe_BLD, k_nope_BLD = torch.split(
            k_BLD, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe_BLNH, k_pe_BL1H = self.rope(q_pe_BLNH, k_pe_BLD.unsqueeze(2), positions)
        idx_q_BLNH = Indexer._hadamard_rotate(
            torch.cat([q_pe_BLNH, q_nope_BLNH], dim=-1)
        )
        idx_k_BL1H = Indexer._hadamard_rotate(
            torch.cat([k_pe_BL1H.squeeze(2), k_nope_BLD], dim=-1)
        )

        idx_w_BLN = self.weights_proj(x_BLD) * (self.n_heads**-0.5)
        idx_w_BLN = idx_w_BLN * (self.head_dim**-0.5)

        return idx_q_BLNH, idx_k_BL1H, idx_w_BLN

    @staticmethod
    def select(
        idx_q_BLNH: torch.Tensor,
        idx_k_BL1H: torch.Tensor,
        idx_w_BLN: torch.Tensor,
        dense_mask: torch.Tensor,
        index_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k KV positions per query from indexer scores.

        Args:
            dense_mask: Dense [B, 1, Lq, Lkv] bool mask (True = attendable),
                materialized once per step in ``get_attention_masks`` and
                shared by all layers.
        """
        Lkv = idx_k_BL1H.shape[1]

        scores_BLqHLkv = torch.relu(
            torch.einsum("blhd,bsd->blhs", idx_q_BLNH.float(), idx_k_BL1H.float())
        )
        index_scores_BLqLkv = (scores_BLqHLkv * idx_w_BLN.unsqueeze(-1).float()).sum(
            dim=2
        )
        index_scores_BLqLkv = index_scores_BLqLkv.where(
            dense_mask.squeeze(1), float("-inf")
        )

        k = min(index_topk, Lkv)
        topk_scores_BLqK, topk_indices_BLqK = index_scores_BLqLkv.topk(k, dim=-1)
        return (
            topk_indices_BLqK.where(topk_scores_BLqK.isfinite(), -1),
            index_scores_BLqLkv,
        )

    @staticmethod
    def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d, device=x.device, dtype=x.dtype)
        return F.linear(x, H) * (d**-0.5)


def _build_selected_bm(
    topk_indices_BLqK: torch.Tensor,
    block_size: tuple[int, int],
    kv_len: int,
) -> BlockMask:
    B, Lq, K = topk_indices_BLqK.shape
    BQ, BK = block_size
    assert Lq % BQ == 0, f"Lq ({Lq}) must be divisible by BQ ({BQ})"

    nQ = Lq // BQ
    nKV = (kv_len + BK - 1) // BK

    # Token indices -> block indices, aggregated to Q-block granularity
    block_kv = (topk_indices_BLqK // BK).reshape(B, nQ, BQ * K)
    valid = (topk_indices_BLqK >= 0).reshape(B, nQ, BQ * K)

    # Dense block-level mask [B, 1, nQ, nKV]: which KV blocks are active?
    bm = torch.zeros(B, 1, nQ, nKV, dtype=torch.int32, device=topk_indices_BLqK.device)
    bm[:, 0].scatter_add_(-1, block_kv.clamp(min=0), valid.to(torch.int32))
    bm = (bm > 0).to(torch.int32)

    # Ordered format matching _dense_to_ordered
    kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
    kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(torch.int32)

    # mask_mod captures topk_indices directly for partial-block masking
    def mask_mod(b, h, q_idx, k_idx):
        return (topk_indices_BLqK[b, q_idx] == k_idx).any(dim=-1)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=(BQ, BK),
        mask_mod=mask_mod,
        seq_lengths=(Lq, kv_len),
    )


class BlockMaskHandler(BaseMaskHandler):
    """Post-processes BlockMask into DSA dict with precomputed dense_mask."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseMaskHandler.Config):
        pass

    def post_process(self, masks):
        assert isinstance(masks, BlockMask), f"expected BlockMask, got {type(masks)}"
        B = masks.kv_num_blocks.shape[0]
        seq_len = masks.seq_lengths[0]
        device = masks.kv_num_blocks.device
        dense_mask = create_mask(
            masks.mask_mod,
            B,
            1,
            seq_len,
            seq_len,
            device=device,
        )
        return {"block_mask": masks, "dense_mask": dense_mask}


class SparseIndexerLoss(LoggedAuxLoss):
    """Indexer alignment loss module with gradient injection.

    Gathers Q and K at the topk_indices positions to avoid materializing
    the full dense ``[B, Lq, Lkv]`` attention matrix.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LoggedAuxLoss.Config):
        coeff: float = 1.0
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BL1H: torch.Tensor,
        scale: float,
        topk_indices_BLqK: torch.Tensor,
        index_scores_BLqLkv: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        B, Lq, K = topk_indices_BLqK.shape

        k_sqz = k_BL1H.float().squeeze(2)
        b_idx = torch.arange(B, device=topk_indices_BLqK.device)[:, None, None]
        k_gathered = k_sqz[b_idx, topk_indices_BLqK.clamp(min=0)]

        logits = torch.einsum("blhd,blkd->blhk", q_BLNH.float(), k_gathered)
        logits = logits * scale
        logits = logits.masked_fill((topk_indices_BLqK < 0).unsqueeze(2), float("-inf"))
        p = F.softmax(logits, dim=-1).mean(dim=2)

        scores_BLqK = index_scores_BLqLkv.gather(-1, topk_indices_BLqK.clamp(min=0))
        scores_BLqK = scores_BLqK.masked_fill(topk_indices_BLqK < 0, float("-inf"))
        q_pred = F.softmax(scores_BLqK, dim=-1)

        # This magic number matches that of NPU kernel.
        eps = 1e-8
        kl_loss = (p * ((p + eps).log() - (q_pred + eps).log())).sum(dim=-1).mean()
        return self.inject(carrier, kl_loss)


class SparseInnerAttention(FlexAttention):
    """Sparse attention core for DSA, with separated nope/rope support."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        index_topk: int
        indexer_loss: SparseIndexerLoss.Config = field(
            default_factory=SparseIndexerLoss.Config
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.index_topk = config.index_topk
        self.indexer_loss = config.indexer_loss.build()

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
        attention_masks: dict[str, BlockMask | torch.Tensor],
        scale: float,
    ) -> torch.Tensor:
        q_BLNH = torch.cat([q_nope_BLNH, q_rope_BLNH], dim=-1)
        k_BL1H = torch.cat([k_nope_BL1H, k_rope_BL1H], dim=-1)
        v_BL1H = k_nope_BL1H

        block_mask = attention_masks["block_mask"]
        dense_mask = attention_masks["dense_mask"]
        assert isinstance(block_mask, BlockMask)
        assert isinstance(dense_mask, torch.Tensor)

        topk_indices_BLqK, index_scores_BLqLkv = Indexer.select(
            idx_q_BLNH,
            idx_k_BL1H,
            idx_w_BLN,
            dense_mask,
            self.index_topk,
        )

        with spmd.no_typecheck():
            selected_bm = _build_selected_bm(
                topk_indices_BLqK, block_mask.BLOCK_SIZE, k_BL1H.shape[1]
            )

        output_BLNH = super().forward(
            q_BLNH,
            k_BL1H,
            v_BL1H,
            attention_masks=selected_bm,
            scale=scale,
            enable_gqa=True,
        )
        if self.training:
            output_BLNH = self.indexer_loss(
                q_BLNH.detach(),
                k_BL1H.detach(),
                scale,
                topk_indices_BLqK,
                index_scores_BLqLkv,
                carrier=output_BLNH,
            )
        return output_BLNH


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

    def forward(
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

        q_nope_BLNH, q_pe_BLNH = torch.split(
            q_BLNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv_BLD = self.wkv_a(x_BLD)
        kv_nope_BL1H, k_pe_BL1H = torch.split(
            kv_BLD.unsqueeze(2), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_nope_BL1H = self.kv_norm(kv_nope_BL1H)

        q_pe_BLNH, k_pe_BL1H = self.rope(q_pe_BLNH, k_pe_BL1H, positions)
        q_nope_BLNH = self.w_uk(q_nope_BLNH)

        idx_q_BLNH, idx_k_BL1H, idx_w_BLN = self.indexer(
            x_BLD.detach(), qr_BLD.detach(), positions=positions
        )
        # `idx_k` needs to be redistributed in tp-cp axis, which is not
        # supported by spmd_types for the time being.
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
        wkv_b_3d = wkv_b.view(module.n_heads, -1, module.kv_lora_rank)
        state_dict[f"{prefix}w_uk.weight"] = (
            wkv_b_3d[:, : module.qk_nope_head_dim, :].transpose(-2, -1).contiguous()
        )
        state_dict[f"{prefix}w_uv.weight"] = wkv_b_3d[
            :, module.qk_nope_head_dim :, :
        ].contiguous()

    @staticmethod
    def _merge_wkv_b_on_save(module, state_dict, prefix, local_metadata):
        w_uk = state_dict.pop(f"{prefix}w_uk.weight")
        w_uv = state_dict.pop(f"{prefix}w_uv.weight")
        wkv_b = torch.cat([w_uk.transpose(-2, -1), w_uv], dim=1).reshape(
            -1, module.kv_lora_rank
        )
        state_dict[f"{prefix}wkv_b.weight"] = wkv_b.contiguous()


class DeepSeekV32Model(DeepSeekV3Model):
    """DeepSeek-V3.2 model — identical to V3 but with V3.2 sharding config."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        mask_handler: BaseMaskHandler.Config = field(
            default_factory=BlockMaskHandler.Config
        )

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan_npu_lite.models.deepseek_v3_2.sharding import (
                set_deepseek_v3_2_sharding_config,
            )

            set_deepseek_v3_2_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self._mask_handler = config.mask_handler.build()
