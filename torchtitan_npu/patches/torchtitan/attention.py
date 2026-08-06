# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/ac13e536c84e7f6647b14fa9375c3c8a8a2b8578/torchtitan/models/common/attention.py
# https://github.com/pytorch/torchtitan/blob/ac13e536c84e7f6647b14fa9375c3c8a8a2b8578/torchtitan/models/common/decoder.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Patches block-causal attention to use dataloader document boundaries.

Chat templates may emit EOS tokens between messages, while greedy packing
boundaries are represented by resets in the dataloader's ``positions`` tensor.
Using token values as document markers therefore splits conversations and can
also miss the boundary between two packed samples.  This patch routes the
position resets to Varlen Attention and dense SDPA masks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchtitan.models.common.attention as _titan_attention
import torchtitan.models.common.decoder as _titan_decoder
from torchtitan.models.common.attention import (
    BaseAttention,
    LocalMapInnerAttention,
    ScaledDotProductAttention,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.models.common.attention import AttentionMasksType


# --- BaseAttention.Config.__post_init__ -------------------------------------


def _patched_base_attention_post_init(self) -> None:
    assert self.n_heads > 0, "n_heads must be > 0"
    assert isinstance(self.inner_attention, LocalMapInnerAttention.Config), (
        f"inner_attention must be a LocalMapInnerAttention.Config, got {type(self.inner_attention)}"
    )
    assert self.mask_type in [
        "causal",
        "block_causal",
    ], f"mask_type must be one of ['causal', 'block_causal'], got {self.mask_type}"


BaseAttention.Config.__post_init__ = _patched_base_attention_post_init


# --- ScaledDotProductAttention.forward --------------------------------------


_original_sdpa_forward = ScaledDotProductAttention.forward


def _patched_sdpa_forward(
    self: ScaledDotProductAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    is_causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    attention_masks = kwargs.pop("attention_masks", None)
    if attention_masks is None:
        return _original_sdpa_forward(
            self,
            q,
            k,
            v,
            scale=scale,
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            **kwargs,
        )
    if not isinstance(attention_masks, torch.Tensor):
        raise TypeError(
            f"ScaledDotProductAttention requires a dense Tensor attention mask, got {type(attention_masks).__name__}"
        )
    if attention_masks.ndim != 4 or attention_masks.dtype != torch.bool:
        raise ValueError(
            "ScaledDotProductAttention expects a bool [batch, 1, seq, seq] "
            f"mask, got shape={tuple(attention_masks.shape)} dtype={attention_masks.dtype}"
        )

    # SDPA interprets bool masks as allow-masks.  The NPU op-plugin inverts
    # this representation when forwarding to npu_fusion_attention_v3.
    attention_masks = attention_masks.to(device=q.device)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    with _titan_attention.sdpa_kernel(self.sdpa_backends, set_priority=True):
        out = _titan_attention.F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_masks,
            scale=scale,
            is_causal=False,
            enable_gqa=enable_gqa,
            **kwargs,
        )
    return out.transpose(1, 2)


ScaledDotProductAttention.forward = _patched_sdpa_forward


# --- Decoder.get_attention_masks --------------------------------------------


def _get_position_boundary_attention(config: object):
    """Return the attention implementation if it supports position boundaries.

    Position resets can be converted to a dense boolean mask for SDPA and to
    ``VarlenMetadata`` for Varlen Attention.
    """

    if not isinstance(config, Decoder.Config) or not config.layers:
        return None
    attention = config.layers[0].attention
    inner_attention = attention.inner_attention
    if attention.mask_type != "block_causal" or not isinstance(
        inner_attention, (ScaledDotProductAttention.Config, VarlenAttention.Config)
    ):
        return None
    return inner_attention


def _document_starts(positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim != 2 or positions.numel() == 0:
        raise ValueError(f"positions must have non-empty shape [batch, seq], got {tuple(positions.shape)}")

    starts = positions.eq(0)
    if not bool(starts[:, 0].all().item()):
        raise ValueError("each packed sequence must start with position 0")
    return starts


def _dense_mask_from_starts(starts: torch.Tensor) -> torch.Tensor:
    document_ids = starts.to(torch.int64).cumsum(dim=1)
    token_indices = torch.arange(starts.shape[1], device=starts.device)
    causal = token_indices[:, None] >= token_indices[None, :]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    return (same_document & causal).unsqueeze(1)


def _varlen_metadata_from_starts(starts: torch.Tensor) -> VarlenMetadata:
    sequence_starts = starts.flatten().nonzero(as_tuple=True)[0].cpu()
    cu_seq = torch.cat((sequence_starts, sequence_starts.new_tensor([starts.numel()])))
    max_seq_len = int(torch.diff(cu_seq).max().item())
    return VarlenMetadata(
        cu_seq_q=cu_seq,
        cu_seq_k=cu_seq,
        max_q=max_seq_len,
        max_k=max_seq_len,
    )


_original_decoder_get_attention_masks = Decoder.get_attention_masks


def _patched_decoder_get_attention_masks(
    self: Decoder,
    input_batch: torch.Tensor,
    tokenizer: BaseTokenizer,
    extra_inputs: dict[str, torch.Tensor] | None = None,
    positions: torch.Tensor | None = None,
) -> AttentionMasksType | torch.Tensor:
    inner_attention = _get_position_boundary_attention(self.config)
    if inner_attention is not None and positions is not None:
        if positions.shape != input_batch.shape:
            raise ValueError(
                f"positions shape {tuple(positions.shape)} must match input batch shape {tuple(input_batch.shape)}"
            )
        starts = _document_starts(positions)
        if isinstance(inner_attention, ScaledDotProductAttention.Config):
            return _dense_mask_from_starts(starts)
        if isinstance(inner_attention, VarlenAttention.Config):
            return _varlen_metadata_from_starts(starts)

    if isinstance(inner_attention, ScaledDotProductAttention.Config):
        raise ValueError("block-causal SDPA attention requires dataloader positions")

    return _original_decoder_get_attention_masks(
        self,
        input_batch=input_batch,
        tokenizer=tokenizer,
        extra_inputs=extra_inputs,
    )


_titan_decoder.Decoder.get_attention_masks = _patched_decoder_get_attention_masks  # pyrefly: ignore [bad-assignment]

logger.info("[Patch] Enabled position-based block-causal masks for SDPA/Varlen attention")
