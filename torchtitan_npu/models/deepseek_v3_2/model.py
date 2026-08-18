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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.nn.attention.flex_attention import BlockMask, create_mask
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.attention import AttentionMasksType, FlexAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v3.model import (
    Attention as V3Attention,
)
from torchtitan.models.deepseek_v3.model import (
    DeepSeekV3Model,
)
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.metadata_extension import MetadataExtension
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import LoggedAuxLoss
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear


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
        dense_mask: torch.Tensor,
        index_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k KV positions per query from indexer scores.

        Args:
            dense_mask: Dense ``[B, 1, Lq, Lkv]`` Boolean mask, where ``True``
                marks an attendable position. It is materialized once per step
                and shared by all layers.
        """
        Lkv = idx_k_BL1H.shape[1]

        scores_BLqHLkv = torch.relu(torch.einsum("blhd,bsd->blhs", idx_q_BLNH.float(), idx_k_BL1H.float()))
        index_scores_BLqLkv = (scores_BLqHLkv * idx_w_BLN.unsqueeze(-1).float()).sum(dim=2)
        index_scores_BLqLkv = index_scores_BLqLkv.where(dense_mask.squeeze(1), float("-inf"))

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

    # Convert token indices to KV-block indices, then aggregate all selected
    # blocks within each query block.
    block_kv = (topk_indices_BLqK // BK).reshape(B, nQ, BQ * K)
    valid = (topk_indices_BLqK >= 0).reshape(B, nQ, BQ * K)

    # Dense block-level mask: ``[B, 1, nQ, nKV]``.
    bm = torch.zeros(B, 1, nQ, nKV, dtype=torch.int32, device=topk_indices_BLqK.device)
    bm[:, 0].scatter_add_(-1, block_kv.clamp(min=0), valid.to(torch.int32))
    bm = (bm > 0).to(torch.int32)

    # Convert the dense mask to the ordered block representation.
    kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
    kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(torch.int32)

    # Preserve token-level selection inside partially selected KV blocks.
    def mask_mod(b, h, q_idx, k_idx):
        return (topk_indices_BLqK[b, q_idx] == k_idx).any(dim=-1)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=(BQ, BK),
        mask_mod=mask_mod,
        seq_lengths=(Lq, kv_len),
    )


class SparseIndexerLoss(LoggedAuxLoss):
    """Indexer alignment loss with gradient injection.

    Only top-k Q/K pairs are gathered, avoiding a dense
    ``[B, Lq, Lkv]`` attention tensor.
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
        B, _Lq, _K = topk_indices_BLqK.shape

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

        # Use the same epsilon as the fused NPU loss kernel for parity.
        eps = 1e-8
        kl_loss = (p * ((p + eps).log() - (q_pred + eps).log())).sum(dim=-1).mean()
        return self.inject(carrier, kl_loss)


class SparseInnerAttention(FlexAttention):
    """Sparse attention core for DSA, with separated nope/rope support."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        index_topk: int
        indexer_loss: SparseIndexerLoss.Config = field(default_factory=SparseIndexerLoss.Config)

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
            selected_bm = _build_selected_bm(topk_indices_BLqK, block_mask.BLOCK_SIZE, k_BL1H.shape[1])

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
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        metadata_extension: MetadataExtension.Config = field(default_factory=MetadataExtension.Config)

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan_npu.models.deepseek_v3_2.sharding import (
                set_deepseek_v3_2_sharding_config,
            )

            set_deepseek_v3_2_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self._metadata_extension = config.metadata_extension.build()

    def build_attention_masks(
        self,
        inputs,
        labels,
        extra_kwargs,
        *,
        cp_mesh: DeviceMesh | None = None,
        load_balancer_type: str | None = None,
    ):
        """The model-owned per-batch metadata construction (the single
        overridable mask-handling seam).

        The core mask comes from the (flex- or varlen-typed) inner attention;
        under context parallel the generic ``prepare_context_parallel_input``
        shards it.  The DSA block mask + the shared dense attendability mask
        are derived from a flex ``BlockMask``; a varlen mask passes through
        to the ``metadata_extension`` (the AscendC TND conversion)."""
        positions = extra_kwargs.get("positions")
        masks = super().get_attention_masks(positions=positions)
        if cp_mesh is not None:
            # ``prepare_context_parallel_input`` shards ``attention_masks``
            # only when the key is already present (it never adds it), so the
            # built masks must be handed over before the CP shard call.
            extra_kwargs["attention_masks"] = masks
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                cp_mesh,
                inputs.device,
                load_balancer_type,
            )
            masks = extra_kwargs["attention_masks"]
        attention_masks = self._build_block_mask(masks) if isinstance(masks, BlockMask) else masks
        if self._metadata_extension is not None:
            attention_masks = self._metadata_extension(attention_masks)
        extra_kwargs["attention_masks"] = attention_masks
        return inputs, labels, extra_kwargs

    @staticmethod
    def _build_block_mask(masks: BlockMask) -> dict:
        """The DSA block mask and its shared dense attendability mask."""
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
