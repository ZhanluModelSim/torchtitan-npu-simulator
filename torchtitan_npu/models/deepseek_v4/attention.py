# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    BaseAttention,
    create_attention_mask,
    FlexAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.aux_loss import (
    LoggedAuxLoss,
)
from torchtitan_npu.patches.torchtitan.models.common.rope import SingleComplexRoPE

from .compressor import Compressor, Indexer, IndexSelection

def _assert_spmd_attention_type(tensor, *, tp):
    if get_spmd_backend() == "spmd_types":
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )


def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    device,
) -> torch.Tensor:
    window = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (base - window + 1).clamp(0) + torch.arange(window, device=device)
    idxs = torch.where(idxs > base, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs(
    compress_ratio: int,
    bsz: int,
    seqlen: int,
    device,
) -> torch.Tensor:
    compress_len = seqlen // compress_ratio
    if compress_len == 0:
        return torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)

    idxs = torch.arange(compress_len, device=device).repeat(seqlen, 1)
    causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
    causal_limit = causal_limit // compress_ratio
    idxs = torch.where(idxs >= causal_limit, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


class DSAIndexerAuxLoss(LoggedAuxLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(LoggedAuxLoss.Config):
        num_heads: int
        softmax_scale: float
        window_size: int
        coeff: float = 1.0
        eps: float = 1e-10

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads
        self.softmax_scale = config.softmax_scale
        self.window_size = config.window_size
        self.eps = config.eps

    def _selected_main_attn_dist(
        self,
        q,
        kv_compress,
        compress_topk_idxs,
        attn_lse,
    ) -> torch.Tensor:
        _, seqlen, _, head_dim = q.size()
        gather_idxs = compress_topk_idxs.clamp(min=0)
        selected_kv = torch.gather(
            kv_compress.unsqueeze(1).expand(-1, seqlen, -1, -1),
            dim=2,
            index=gather_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim),
        )

        selected_score = (
            torch.einsum("bshd,bskd->bhsk", q, selected_kv) * self.softmax_scale
        )
        selected_prob = torch.exp(
            selected_score.float() - attn_lse.transpose(1, 2).unsqueeze(-1).float()
        )
        selected_prob = selected_prob.masked_fill(
            compress_topk_idxs.unsqueeze(1) < 0, 0.0
        )
        return selected_prob.sum(dim=1) / self.num_heads

    def _indexer_loss(
        self,
        selected_main_attn_dist,
        index_score,
        compress_topk_idxs,
    ) -> torch.Tensor:
        selected_main_attn_dist = selected_main_attn_dist.float().clamp_min(0)
        target_sum = selected_main_attn_dist.sum(dim=-1, keepdim=True)
        valid_target = target_sum > self.eps

        index_score = torch.where(
            valid_target,
            index_score,
            torch.zeros_like(index_score),
        )
        index_score = F.log_softmax(index_score, dim=-1, dtype=torch.float32)

        selected_main_attn_dist = selected_main_attn_dist / target_sum.clamp_min(
            self.eps
        )
        positive_target = selected_main_attn_dist > 0
        index_score = torch.where(
            positive_target,
            index_score,
            torch.zeros_like(index_score),
        )
        log_selected_main_attn_dist = selected_main_attn_dist.clamp_min(
            self.eps
        ).log()
        loss = (
            selected_main_attn_dist * (log_selected_main_attn_dist - index_score)
        ).sum(dim=-1)
        loss = (target_sum.squeeze(-1) * loss).mean()
        return loss * self.softmax_scale

    def forward(
        self,
        carrier,
        q,
        kv_compress,
        compress_topk_idxs,
        index_score,
        attn_lse,
    ):
        if index_score.numel() == 0:
            return carrier
        selected_main_attn_dist = self._selected_main_attn_dist(
            q,
            kv_compress,
            compress_topk_idxs,
            attn_lse,
        )
        loss = self._indexer_loss(
            selected_main_attn_dist,
            index_score,
            compress_topk_idxs,
        )
        if self.global_batch_size is None:
            raise RuntimeError("DSAIndexerAuxLoss requires global_batch_size.")
        return self.inject(carrier, loss * self.global_batch_size)


class DSAFlexAttention(FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_selection: IndexSelection.Config | None = None
        indexer_aux_loss: DSAIndexerAuxLoss.Config | None = None
        return_lse: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.block_size = config.block_size
        self.return_lse = config.return_lse or config.indexer_aux_loss is not None
        if config.index_selection is not None:
            self.index_selection = config.index_selection.build()
        if config.indexer_aux_loss is not None:
            self.indexer_aux_loss = config.indexer_aux_loss.build()

    def _select_compressed(
        self,
        query_states,
        kv_compress,
        q_indexer,
        k_indexer,
        index_weights,
        *,
        attention_masks=None,
    ):
        bsz, seqlen, _, _ = query_states.size()
        if self.compress_ratio == 4:
            if not hasattr(self, "index_selection"):
                raise RuntimeError(
                    "DSAFlexAttention requires index_selection when compress_ratio=4."
                )
            if q_indexer is None or k_indexer is None or index_weights is None:
                raise ValueError(
                    "DSAFlexAttention requires q_indexer, k_indexer, and "
                    "index_weights when compress_ratio=4."
                )
            return self.index_selection(
                q_indexer,
                k_indexer,
                index_weights,
                attention_masks=attention_masks,
            )

        if self.compress_ratio > 1:
            if kv_compress is None:
                raise ValueError(
                    "DSAFlexAttention requires compressed KV when compress_ratio>1."
                )
            return (
                get_compress_topk_idxs(
                    self.compress_ratio,
                    bsz,
                    seqlen,
                    query_states.device,
                ),
                None,
            )

        return None, None

    def _prepare_sparse_inputs(
        self,
        query_states,
        kv_states,
        kv_compress,
        q_indexer,
        k_indexer,
        index_weights,
        *,
        attention_masks=None,
    ):
        bsz, seqlen, _, _ = query_states.size()
        window_topk_idxs = get_window_topk_idxs(
            self.window_size,
            bsz,
            seqlen,
            query_states.device,
        )
        compress_topk_idxs, index_score = self._select_compressed(
            query_states,
            kv_compress,
            q_indexer,
            k_indexer,
            index_weights,
            attention_masks=attention_masks,
        )

        if (
            compress_topk_idxs is not None
            and kv_compress is not None
            and compress_topk_idxs.size(-1) > kv_compress.size(1)
        ):
            compact_topk = kv_compress.size(1)
            compress_topk_idxs = compress_topk_idxs[..., :compact_topk]
            if index_score is not None:
                index_score = index_score[..., :compact_topk]

        if kv_compress is None or kv_compress.numel() == 0:
            return kv_states, window_topk_idxs, compress_topk_idxs, index_score

        if compress_topk_idxs is None:
            raise RuntimeError(
                "DSAFlexAttention received compressed KV without compressed indices."
            )
        kv_offset = kv_states.size(1)
        attention_compress_idxs = torch.where(
            compress_topk_idxs < 0,
            compress_topk_idxs,
            compress_topk_idxs + kv_offset,
        )
        topk_idxs = torch.cat([window_topk_idxs, attention_compress_idxs], dim=-1)
        return (
            torch.cat([kv_states, kv_compress], dim=1),
            topk_idxs,
            compress_topk_idxs,
            index_score,
        )

    def _apply_indexer_aux_loss(
        self,
        carrier,
        query_states,
        kv_compress,
        compress_topk_idxs,
        index_score,
        attn_lse,
    ):
        if (
            not self.training
            or not hasattr(self, "indexer_aux_loss")
            or kv_compress is None
            or compress_topk_idxs is None
            or index_score is None
            or attn_lse is None
        ):
            return carrier
        return self.indexer_aux_loss(
            carrier,
            query_states.detach(),
            kv_compress.detach(),
            compress_topk_idxs,
            index_score,
            attn_lse.detach(),
        )

    def _create_topk_mask(
        self,
        *,
        topk_idxs: torch.Tensor,
        bsz: int,
        seqlen: int,
        kv_len: int,
        device,
    ):
        sink_idx = kv_len
        topk = topk_idxs.size(-1)

        def v4_sparse_mask_mod(b, h, q_idx, kv_idx):
            selected = kv_idx < 0
            for idx in range(topk):
                selected = selected | (topk_idxs[b, q_idx, idx] == kv_idx)
            is_sink = kv_idx == sink_idx
            return selected | is_sink

        return create_attention_mask(
            v4_sparse_mask_mod,
            bsz,
            None,
            seqlen,
            kv_len + 1,
            device=device,
            BLOCK_SIZE=self.block_size,
            separate_full_blocks=False,
        )

    def forward(
        self,
        query_states,
        kv_states,
        kv_compress,
        attn_sink,
        q_indexer=None,
        k_indexer=None,
        index_weights=None,
        *,
        attention_masks=None,
    ):
        bsz, seqlen, _, head_dim = query_states.size()
        (
            kv_states,
            topk_idxs,
            compress_topk_idxs,
            index_score,
        ) = self._prepare_sparse_inputs(
            query_states,
            kv_states,
            kv_compress,
            q_indexer,
            k_indexer,
            index_weights,
            attention_masks=attention_masks,
        )
        kv_len = kv_states.size(1)

        sink_kv = kv_states.new_zeros((bsz, 1, head_dim))
        kv_states = torch.cat([kv_states, sink_kv], dim=1)
        key_value_states = kv_states.unsqueeze(2)

        block_mask = self._create_topk_mask(
            topk_idxs=topk_idxs,
            bsz=bsz,
            seqlen=seqlen,
            kv_len=kv_len,
            device=query_states.device,
        )
        sink_idx = kv_len

        def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_idx == sink_idx, attn_sink[h], score)

        def maybe_return_lse(out, lse):
            return out, lse

        attn_out = super().forward(
            query_states,
            key_value_states,
            key_value_states,
            attention_masks=block_mask,
            score_mod=v4_sink_score_mod,
            scale=self.softmax_scale,
            enable_gqa=True,
            out_transform=maybe_return_lse if self.return_lse else None,
        )
        if self.return_lse:
            output, attn_lse = attn_out
        else:
            output, attn_lse = attn_out, None
        return self._apply_indexer_aux_loss(
            output,
            query_states,
            kv_compress,
            compress_topk_idxs,
            index_score,
            attn_lse,
        )


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        single_rope: SingleComplexRoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        window_size: int = 128
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Declare submodule configs as fields so sharding can be assigned before
        # the modules are built.
        wq_a: Linear.Config | None = None
        q_norm: RMSNorm.Config | None = None
        wq_b: Linear.Config | None = None
        wkv: Linear.Config | None = None
        kv_norm: RMSNorm.Config | None = None
        wo_a: Linear.Config | None = None
        wo_b: Linear.Config | None = None
        attn_sink: Linear.Config | None = None

        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.window_size = cfg.window_size
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers
        self.rope = cfg.rope.build()
        self.single_rope = cfg.single_rope.build()

        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = cfg.attn_sink.build().float()

        if cfg.compressor is not None:
            self.compressor = cfg.compressor.build()
        if cfg.indexer is not None:
            self.indexer = cfg.indexer.build()
        if cfg.compressor_128 is not None:
            self.compressor_128 = cfg.compressor_128.build()

        self.inner_attention = cfg.inner_attention.build()
        self._dsa_loss_tracker = None

    def set_dsa_loss_tracker(self, tracker):
        self._dsa_loss_tracker = tracker

    def forward(self, x, attention_masks=None, positions=None):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        _assert_spmd_attention_type(qr, tp=spmd.R)
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(bsz, seqlen, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(2))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        _assert_spmd_attention_type(kv, tp=spmd.R)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(2)], dim=-1)
        _assert_spmd_attention_type(q, tp=spmd.S(2))
        _assert_spmd_attention_type(kv, tp=spmd.R)

        kv_compress = None
        q_indexer = k_indexer = index_weights = None

        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            q_indexer, k_indexer, index_weights = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_masks=attention_masks,
            )

        if self.compress_ratio == 4:
            kv_compress = self.compressor(
                x,
                positions=positions,
                attention_masks=attention_masks,
            )
        elif self.compress_ratio > 1:
            kv_compress = self.compressor_128(
                x,
                positions=positions,
                attention_masks=attention_masks,
            )

        # Keep the inner-attention positional contract tensor-only. TP
        # ``local_map`` requires a placement for every positional input, even
        # when a layer does not use compressed KV or the learned indexer.
        if kv_compress is None:
            kv_compress = kv.new_empty((bsz, 0, self.head_dim))
        if q_indexer is None:
            q_indexer = x.new_empty((bsz, seqlen, 0, 0))
            k_indexer = x.new_empty((bsz, 0, 0))
            index_weights = x.new_empty((bsz, seqlen, 0))
        _assert_spmd_attention_type(kv_compress, tp=spmd.R)
        _assert_spmd_attention_type(q_indexer, tp=spmd.R)
        _assert_spmd_attention_type(k_indexer, tp=spmd.R)
        _assert_spmd_attention_type(index_weights, tp=spmd.R)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        o = self.inner_attention(
            q,
            kv,
            kv_compress,
            attn_sink_param,
            q_indexer,
            k_indexer,
            index_weights,
            attention_masks=attention_masks,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.single_rope(o_rope, positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)
        _assert_spmd_attention_type(o, tp=spmd.S(2))

        with spmd.local():
            n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
            o = o.view(bsz, seqlen, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(2))
            # ``wo_a`` stores grouped LoRA-A weights in a ``Linear`` module,
            # but the grouped projection is computed directly with einsum.
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        with spmd.local():
            o = o.reshape(bsz, seqlen, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(2))
        return self.wo_b(o)
