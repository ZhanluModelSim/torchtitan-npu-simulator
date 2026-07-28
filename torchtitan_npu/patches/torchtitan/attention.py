# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/ac13e536c84e7f6647b14fa9375c3c8a8a2b8578/torchtitan/models/common/attention.py
# https://github.com/pytorch/torchtitan/blob/ac13e536c84e7f6647b14fa9375c3c8a8a2b8578/torchtitan/models/common/decoder.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Patches for torchtitan attention and dense block-causal SDPA support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchtitan.models.common.attention as _titan_attention
import torchtitan.models.common.decoder as _titan_decoder
from torchtitan.models.common.attention import (
    BaseAttention,
    LocalMapInnerAttention,
    ScaledDotProductAttention,
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


def _is_block_causal_sdpa_config(config: object) -> bool:
    if not isinstance(config, Decoder.Config) or not config.layers:
        return False
    attention = config.layers[0].attention
    return (
        isinstance(attention.inner_attention, ScaledDotProductAttention.Config)
        and attention.mask_type == "block_causal"
    )


def build_block_causal_sdpa_mask(
    input_batch: torch.Tensor,
    eos_id: int,
) -> torch.Tensor:
    """Build a dense ``[batch, 1, seq, seq]`` block-causal allow-mask.

    ``True`` entries are positions visible to SDPA.  An EOS token remains in
    the document it terminates; the following token starts a new document.
    This is the same boundary convention used by torchtitan's flex/varlen
    document-mask helpers.
    """

    if input_batch.ndim != 2:
        raise ValueError(f"input_batch must have shape [batch, seq], got {tuple(input_batch.shape)}")

    eos = input_batch.eq(eos_id)
    # Match torchtitan's document-mask convention by terminating the physical
    # sequence even when its final token is not EOS.  The current token remains
    # in its document; this only gives the terminal position a stable boundary
    # for document-id construction.
    eos = eos.clone()
    eos[:, -1] = True
    document_ids = torch.cumsum(eos.to(torch.int64), dim=1) - eos.to(torch.int64)

    seq_len = input_batch.shape[1]
    positions = torch.arange(seq_len, device=input_batch.device)
    causal = positions[:, None] >= positions[None, :]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    return (same_document & causal).unsqueeze(1)


_original_decoder_get_attention_masks = Decoder.get_attention_masks


def _patched_decoder_get_attention_masks(
    self: Decoder,
    input_batch: torch.Tensor,
    tokenizer: BaseTokenizer,
    extra_inputs: dict[str, torch.Tensor] | None = None,
) -> AttentionMasksType | torch.Tensor:
    if _is_block_causal_sdpa_config(self.config):
        if tokenizer.eos_id is None:
            raise ValueError("tokenizer.eos_id is required for block-causal SDPA")
        return build_block_causal_sdpa_mask(input_batch, tokenizer.eos_id)
    return _original_decoder_get_attention_masks(
        self,
        input_batch=input_batch,
        tokenizer=tokenizer,
        extra_inputs=extra_inputs,
    )


_titan_decoder.Decoder.get_attention_masks = _patched_decoder_get_attention_masks  # pyrefly: ignore [bad-assignment]

logger.info("[Patch] Enabled dense block-causal masks for SDPA")
