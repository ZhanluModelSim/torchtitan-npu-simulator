# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: provide the DeepSeek-V4 numerical reference.

RMSNorm, MoE, and DSA sparse attention match the ``dsv4-infer-npu`` baseline.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm

from torchtitan_npu.models.deepseek_v4.attention import (
    DSAFlexAttention,
    get_window_topk_idxs,
)
from torchtitan_npu.models.deepseek_v4.moe import DeepSeekV4MoE
from torchtitan_npu.models.deepseek_v4.packed import (
    DSV4PackedMetadata,
    compact_compressed_tensor,
    compact_token_tensor,
    restore_token_tensor,
)
from torchtitan_npu.override.deepseek_v4.varlen_dsa import (
    DSAVarlenAttention,
    derive_varlen_dsa,
)

# Bound the ``[B, M, topk, D]`` gather used by sparse attention.
_ATTN_CHUNK = int(os.environ.get("TTNPU_DSA_ATTN_CHUNK", "256"))


class GoldenRMSNorm(RMSNorm):
    """RMSNorm matching the inference baseline rounding."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.weight is not None and self.weight.dtype != torch.float32:
            self.weight.data = self.weight.data.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight is None:
            return x.to(dtype)
        return (self.weight.float() * x).to(dtype)


@override(
    target=RMSNorm.Config,
    description="DSV4 golden RMSNorm (baseline-exact FP32 recipe)",
)
def rms_norm_golden(cfg: RMSNorm.Config) -> GoldenRMSNorm.Config:
    return derive(cfg, GoldenRMSNorm.Config)


class GoldenDSV4MoE(DeepSeekV4MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4MoE.Config):
        # The baseline uses a 10.0 SwiGLU limit; zero disables the clamp.
        swiglu_limit: float = 10.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def _expert_ffn(self, x, w1, w2, w3, weights=None):
        # Match baseline precision: BF16 matmuls and FP32 activation/routing.
        dtype = x.dtype
        gate = F.linear(x, w1).float()
        up = F.linear(x, w3).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return F.linear(h.to(dtype), w2)

    def forward(self, x_BLD: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x_BLD.size()
        x_TD = x_BLD.reshape(-1, shape[-1])

        topk_scores_BLK, topk_expert_ids_BLK, scores_BLE = self.router(
            x_BLD, self.expert_bias_E, input_ids=input_ids
        )
        routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
            -1, topk_expert_ids_BLK, True
        )
        with torch.no_grad():
            self.tokens_per_expert_E.add_(routing_map_BLE.sum(dim=(0, 1)))

        top_k = topk_scores_BLK.size(-1)
        weights_TK = topk_scores_BLK.reshape(-1, top_k).float()
        indices_TK = topk_expert_ids_BLK.reshape(-1, top_k)

        inner_experts = self.routed_experts.inner_experts
        w1_EFD = inner_experts.w1_EFD
        w2_EDF = inner_experts.w2_EDF
        w3_EFD = inner_experts.w3_EFD

        y_TD = torch.zeros_like(x_TD, dtype=torch.float32)
        if torch.compiler.is_compiling():
            # Keep a fixed token dimension for fullgraph capture. Each token has
            # at most one slot per expert, so masked weights can be summed.
            for e in range(inner_experts.num_experts):
                expert_weights_T1 = torch.where(
                    indices_TK == e,
                    weights_TK,
                    torch.zeros_like(weights_TK),
                ).sum(dim=-1, keepdim=True)
                y_TD += self._expert_ffn(
                    x_TD,
                    w1_EFD[e],
                    w2_EDF[e],
                    w3_EFD[e],
                    expert_weights_T1,
                )
        else:
            for e in range(inner_experts.num_experts):
                idx, top = torch.where(indices_TK == e)
                if idx.numel() == 0:
                    continue
                y_TD[idx] += self._expert_ffn(
                    x_TD[idx],
                    w1_EFD[e],
                    w2_EDF[e],
                    w3_EFD[e],
                    weights_TK[idx, top, None],
                )
        if self.shared_experts is not None:
            se = self.shared_experts
            y_TD += self._expert_ffn(x_TD, se.w1.weight, se.w2.weight, se.w3.weight)
        return y_TD.type_as(x_TD).view(shape)


@override(
    target=DeepSeekV4MoE.Config,
    exact=True,
    description="DSV4 golden MoE (baseline-exact FP32 recipe, single-device)",
)
def dsv4_moe_golden(cfg: DeepSeekV4MoE.Config) -> GoldenDSV4MoE.Config:
    return derive(cfg, GoldenDSV4MoE.Config)


def _sparse_attn_chunk(
    q_BMHD: torch.Tensor,
    kv_BND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    topk_idxs_BMK: torch.Tensor,
    softmax_scale: float,
    *,
    return_lse: bool,
):
    """Match the baseline sparse-attention chunk with a per-head sink."""
    b, m, k = topk_idxs_BMK.shape

    # Map unused ``-1`` slots to row 0, then mask their values.
    valid_BMK = topk_idxs_BMK != -1
    safe_BMK = torch.where(
        valid_BMK, topk_idxs_BMK, torch.zeros_like(topk_idxs_BMK)
    ).long()
    batch_BMK = (
        torch.arange(b, device=kv_BND.device).view(b, 1, 1).expand(-1, m, k)
    )
    kv_BMKD = kv_BND[batch_BMK, safe_BMK, :] * valid_BMK.unsqueeze(-1).to(
        kv_BND.dtype
    )

    scores_BMHK = torch.matmul(q_BMHD.float(), kv_BMKD.float().transpose(-2, -1))
    scores_BMHK = scores_BMHK * softmax_scale
    scores_BMHK = scores_BMHK.masked_fill(~valid_BMK.unsqueeze(-2), -torch.inf)

    # Include the learned per-head sink in the softmax denominator.
    max_BMH1 = scores_BMHK.max(dim=-1, keepdim=True).values
    exp_BMHK = torch.exp(scores_BMHK - max_BMH1)
    sum_BMH1 = exp_BMHK.sum(dim=-1, keepdim=True)
    sink_BMH1 = torch.exp(attn_sink_H.view(1, 1, -1, 1).float() - max_BMH1)
    probs_BMHK = exp_BMHK / (sum_BMH1 + sink_BMH1)

    out_BMHD = torch.matmul(probs_BMHK, kv_BMKD.to(torch.float32))
    if not return_lse:
        return out_BMHD

    # Include the sink in the LSE consumed by the indexer auxiliary loss.
    lse_BMH = (max_BMH1 + torch.log(sum_BMH1 + sink_BMH1)).squeeze(-1)
    return out_BMHD, lse_BMH


def _sparse_attn(
    q_BMHD: torch.Tensor,
    kv_BND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    topk_idxs_BMK: torch.Tensor,
    softmax_scale: float,
    *,
    return_lse: bool,
):
    m = q_BMHD.size(1)
    chunk = _ATTN_CHUNK if _ATTN_CHUNK > 0 else m
    outs, lses = [], []
    for i in range(0, m, chunk):
        res = _sparse_attn_chunk(
            q_BMHD[:, i : i + chunk],
            kv_BND,
            attn_sink_H,
            topk_idxs_BMK[:, i : i + chunk],
            softmax_scale,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = res
            outs.append(out)
            lses.append(lse)
        else:
            outs.append(res)
    out_BMHD = torch.cat(outs, dim=1)
    if return_lse:
        return out_BMHD, torch.cat(lses, dim=1)
    return out_BMHD


class GoldenDSASparseAttention(DSAVarlenAttention):
    """Reference DSA sparse attention over packed varlen metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(DSAVarlenAttention.Config):
        pass

    def _packed_compressed_indices(
        self,
        metadata: DSV4PackedMetadata,
        device,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            width = metadata.container_seq_len // self.compress_ratio
            candidates = torch.arange(width, device=device).expand(
                metadata.total_tokens, -1
            )
            limit = torch.div(
                metadata.token_positions + 1,
                self.compress_ratio,
                rounding_mode="floor",
            )
            return torch.where(candidates < limit.unsqueeze(1), candidates, -1)

        compressed = metadata.compression_for_ratio(self.compress_ratio)
        width = compressed.varlen.max_k
        indices = torch.full(
            (metadata.total_tokens, width),
            -1,
            dtype=torch.int64,
            device=device,
        )
        for document_id, (q_start, q_end) in enumerate(metadata.sequence_ranges):
            c_start, c_end = compressed.sequence_ranges[document_id]
            compressed_len = c_end - c_start
            if compressed_len == 0:
                continue
            local = torch.arange(q_end - q_start, device=device)
            limit = torch.div(
                local + 1, self.compress_ratio, rounding_mode="floor"
            )
            candidates = torch.arange(compressed_len, device=device).expand(
                q_end - q_start, -1
            )
            indices[q_start:q_end, :compressed_len] = torch.where(
                candidates < limit.unsqueeze(1), candidates, -1
            )
        return indices

    def _apply_packed_indexer_loss(
        self,
        carrier,
        query,
        compressed_kv,
        compressed_indices,
        index_score,
        attn_lse,
        metadata,
    ):
        if (
            not self.training
            or not hasattr(self, "indexer_aux_loss")
            or compressed_kv.numel() == 0
            or index_score is None
        ):
            return carrier
        compressed = metadata.compression_for_ratio(self.compress_ratio)
        query_docs = metadata.token_sequence_ids
        starts = torch.searchsorted(compressed.document_ids, query_docs)
        valid = compressed_indices >= 0
        global_indices = (
            starts.unsqueeze(1) + compressed_indices.clamp_min(0)
        ).clamp_max(compressed_kv.shape[0] - 1)
        selected_kv = compressed_kv.index_select(
            0, global_indices.flatten()
        ).view(*global_indices.shape, compressed_kv.shape[-1])
        selected_score = (
            torch.einsum("thd,tkd->thk", query, selected_kv)
            * self.softmax_scale
        )
        selected_prob = torch.exp(
            selected_score.float() - attn_lse.unsqueeze(-1).float()
        )
        selected_prob = selected_prob.masked_fill(~valid.unsqueeze(1), 0.0)
        selected_main_attn_dist = selected_prob.sum(dim=1) / query.shape[1]
        loss = self.indexer_aux_loss._indexer_loss(
            selected_main_attn_dist,
            index_score,
            compressed_indices,
        )
        if self.indexer_aux_loss.global_batch_size is None:
            raise RuntimeError("DSAIndexerAuxLoss requires global_batch_size.")
        return self.indexer_aux_loss.inject(
            carrier,
            loss * self.indexer_aux_loss.global_batch_size,
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
        if not isinstance(attention_masks, DSV4PackedMetadata):
            raise TypeError(
                "GoldenDSASparseAttention requires DSV4PackedMetadata "
                f"attention masks, got {type(attention_masks)}."
            )
        metadata = attention_masks
        query = compact_token_tensor(query_states, metadata)
        original_kv = compact_token_tensor(kv_states, metadata)
        compressed_kv = (
            query.new_empty((0, query.shape[-1]))
            if self.compress_ratio <= 1
            else compact_compressed_tensor(
                kv_compress, metadata, self.compress_ratio
            )
        )

        index_score = None
        if self.compress_ratio == 4:
            outer_indices, outer_scores = self.index_selection(
                q_indexer,
                k_indexer,
                index_weights,
                attention_masks=metadata,
            )
            compressed_indices = compact_token_tensor(outer_indices, metadata)
            index_score = compact_token_tensor(outer_scores, metadata)
        elif self.compress_ratio > 1:
            compressed_indices = self._packed_compressed_indices(
                metadata, query.device
            )
        else:
            compressed_indices = None

        if torch.compiler.is_compiling():
            token_positions = metadata.token_positions.to(torch.int64)
            token_ids = torch.arange(metadata.total_tokens, device=query.device)
            document_starts = token_ids - token_positions
            window = min(self.window_size, metadata.container_seq_len)
            local_starts = (token_positions - window + 1).clamp_min(0)
            local_indices = local_starts.unsqueeze(1) + torch.arange(
                window, device=query.device
            )
            window_indices = torch.where(
                local_indices <= token_positions.unsqueeze(1),
                document_starts.unsqueeze(1) + local_indices,
                -1,
            )

            indices = window_indices
            all_kv = original_kv
            if self.compress_ratio > 1:
                compressed = metadata.compression_for_ratio(self.compress_ratio)
                compressed_starts = torch.searchsorted(
                    compressed.document_ids,
                    metadata.token_sequence_ids,
                )
                compressed_valid = compressed_indices >= 0
                global_compressed = (
                    compressed_starts.unsqueeze(1)
                    + compressed_indices.clamp_min(0)
                    + metadata.total_tokens
                )
                global_compressed = torch.where(
                    compressed_valid,
                    global_compressed,
                    -1,
                )
                indices = torch.cat((indices, global_compressed), dim=-1)
                all_kv = torch.cat((all_kv, compressed_kv), dim=0)

            output, attn_lse = _sparse_attn(
                query.unsqueeze(0),
                all_kv.unsqueeze(0),
                attn_sink,
                indices.unsqueeze(0),
                self.softmax_scale,
                return_lse=True,
            )
            output = output.squeeze(0).to(query.dtype)
            attn_lse = attn_lse.squeeze(0)
            carrier = restore_token_tensor(output, metadata)
            if self.compress_ratio == 4:
                carrier = self._apply_packed_indexer_loss(
                    carrier,
                    query.detach(),
                    compressed_kv.detach(),
                    compressed_indices,
                    index_score,
                    attn_lse.detach(),
                    metadata,
                )
            return carrier

        outputs, lses = [], []
        compressed = (
            None
            if self.compress_ratio <= 1
            else metadata.compression_for_ratio(self.compress_ratio)
        )
        for document_id, (q_start, q_end) in enumerate(metadata.sequence_ranges):
            length = q_end - q_start
            document_query = query[q_start:q_end].unsqueeze(0)
            document_kv = original_kv[q_start:q_end]
            indices = get_window_topk_idxs(
                self.window_size, 1, length, query.device
            )
            if compressed is not None:
                c_start, c_end = compressed.sequence_ranges[document_id]
                document_compressed = compressed_kv[c_start:c_end]
                document_indices = compressed_indices[
                    q_start:q_end, : c_end - c_start
                ]
                document_indices = torch.where(
                    document_indices < 0,
                    document_indices,
                    document_indices + length,
                ).unsqueeze(0)
                indices = torch.cat([indices, document_indices], dim=-1)
                document_kv = torch.cat(
                    [document_kv, document_compressed], dim=0
                )
            result, lse = _sparse_attn(
                document_query,
                document_kv.unsqueeze(0),
                attn_sink,
                indices,
                self.softmax_scale,
                return_lse=True,
            )
            outputs.append(result.squeeze(0))
            lses.append(lse.squeeze(0))

        output = torch.cat(outputs, dim=0).to(query.dtype)
        attn_lse = torch.cat(lses, dim=0)
        carrier = restore_token_tensor(output, metadata)
        if self.compress_ratio == 4:
            carrier = self._apply_packed_indexer_loss(
                carrier,
                query.detach(),
                compressed_kv.detach(),
                compressed_indices,
                index_score,
                attn_lse.detach(),
                metadata,
            )
        return carrier


@override(
    target=DSAFlexAttention.Config,
    exact=True,
    description=(
        "DSV4 sparse attention golden reference "
        "(replaces DSAFlexAttention flex_attention path)"
    ),
)
def dsa_sparse_attention_golden(
    cfg: DSAFlexAttention.Config,
) -> GoldenDSASparseAttention.Config:
    return derive_varlen_dsa(cfg, GoldenDSASparseAttention.Config)
