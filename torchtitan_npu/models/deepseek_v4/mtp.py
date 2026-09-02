# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__all__ = [
    "DeepSeekV4MTPDecoder",
    "DeepSeekV4MTPTransformerBlock",
    "MTPBatch",
    "MTPChunkedLossWrapper",
    "MTPLoss",
    "MTPModelOutput",
    "prepare_mtp_batch",
    "roll_mtp_sequence",
]

from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import spmd_types as spmd
import torch
from torchtitan.components.loss import (
    IGNORE_INDEX,
    BaseLoss,
    ChunkedLossWrapper,
)
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import AttentionMasksType  # noqa: TC002
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear  # noqa: TC002
from torchtitan.models.common.moe import MoE  # noqa: TC002
from torchtitan.models.common.nn_modules import RMSNorm  # noqa: TC002
from torchtitan.models.deepseek_v3.mtp import (
    MTPDecoder,
    MTPLoss,
    roll_mtp_sequence,
)
from torchtitan.protocols.module import ModuleList

from .mhc import HcHead, HcPost, HcPre  # noqa: TC001


class MTPBatch(NamedTuple):
    """Per-depth MTP tensors with depth stored in the last dimension."""

    input_tokens: torch.Tensor
    input_valid_mask: torch.Tensor
    labels: torch.Tensor


class MTPModelOutput(list[torch.Tensor]):
    """DeepSeek-V4 predictions with optional pre-sharded MTP labels."""

    def __init__(
        self,
        predictions: list[torch.Tensor],
        *,
        mtp_labels: torch.Tensor | None = None,
    ) -> None:
        super().__init__(predictions)
        self.mtp_labels = mtp_labels


def prepare_mtp_batch(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor | None,
    num_mtp_layers: int,
) -> MTPBatch:
    """Build DeepSeek-V4 MTP inputs before context-parallel sharding."""
    if num_mtp_layers <= 0:
        raise ValueError(f"num_mtp_layers must be positive, got {num_mtp_layers}.")

    input_tokens = []
    input_valid_masks = []
    mtp_labels = []
    for depth in range(1, num_mtp_layers + 1):
        depth_tokens, valid_mask = roll_mtp_sequence(
            tokens,
            shift=depth,
            positions=positions,
            fill_value=0,
            return_valid_mask=True,
        )
        input_tokens.append(depth_tokens)
        input_valid_masks.append(valid_mask)
        mtp_labels.append(
            cast(
                "torch.Tensor",
                roll_mtp_sequence(
                    labels,
                    shift=depth,
                    positions=positions,
                    fill_value=IGNORE_INDEX,
                ),
            )
        )

    return MTPBatch(
        input_tokens=torch.stack(input_tokens, dim=-1),
        input_valid_mask=torch.stack(input_valid_masks, dim=-1),
        labels=torch.stack(mtp_labels, dim=-1),
    )


def _annotate_mtp_batch(batch: MTPBatch) -> None:
    if get_spmd_backend() != "spmd_types":
        return
    token_layout = {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.R}
    label_layout = {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.I}
    spmd.assert_type(batch.input_tokens, token_layout)
    spmd.assert_type(batch.input_valid_mask, token_layout)
    spmd.assert_type(batch.labels, label_layout)


def _mtp_labels(
    pred: list[torch.Tensor],
    labels: torch.Tensor,
    positions: torch.Tensor | None,
) -> list[torch.Tensor]:
    num_mtp_layers = len(pred) - 1
    if num_mtp_layers <= 0:
        raise ValueError("MTP loss expects main prediction plus at least one MTP prediction.")

    prepared = getattr(pred, "mtp_labels", None)
    if prepared is not None:
        if prepared.shape[-1] != num_mtp_layers:
            raise ValueError(
                "MTP labels depth does not match predictions: "
                f"{prepared.shape[-1]} labels for {num_mtp_layers} predictions."
            )
        return [prepared[..., depth] for depth in range(num_mtp_layers)]

    if positions is None:
        raise ValueError("MTP loss requires positions when labels were not prepared.")
    return [
        cast(
            "torch.Tensor",
            roll_mtp_sequence(
                labels,
                shift=depth,
                positions=positions,
                fill_value=IGNORE_INDEX,
            ),
        )
        for depth in range(1, num_mtp_layers + 1)
    ]


class _ScaledLoss:
    def __init__(self, loss_fn: BaseLoss) -> None:
        self.loss_fn = loss_fn
        self.scale = 1.0

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self.loss_fn(*args, **kwargs)
        return loss * self.scale, metrics


class MTPChunkedLossWrapper(ChunkedLossWrapper):
    """Apply chunked loss to DeepSeek-V4 main and MTP predictions."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        mtp_scale: float = 0.3

    def __init__(self, config: Config, *, compile_config=None):
        super().__init__(config, compile_config=compile_config)
        self.mtp_scale = config.mtp_scale
        self._scaled_loss = _ScaledLoss(self.loss_fn)
        self.loss_fn = self._scaled_loss  # pyrefly: ignore [bad-assignment]

    def __call__(
        self,
        pred: list[torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(pred, list):
            raise ValueError("MTPChunkedLossWrapper expects main hidden states followed by one tensor per MTP layer.")

        positions = loss_inputs.pop("positions", None)
        mtp_labels = _mtp_labels(pred, labels, positions)
        branch_scale = self.mtp_scale / len(mtp_labels)

        self._scaled_loss.scale = 1.0
        total_loss, metrics = super().__call__(
            pred[0],
            labels,
            global_valid_tokens,
            **loss_inputs,
        )
        for hidden, depth_labels in zip(pred[1:], mtp_labels, strict=True):
            self._scaled_loss.scale = branch_scale
            depth_loss, depth_metrics = super().__call__(
                hidden,
                depth_labels,
                global_valid_tokens,
                **loss_inputs,
            )
            total_loss = total_loss + depth_loss
            metrics = self._combine_chunk_metrics(metrics, depth_metrics)
        return total_loss, metrics


class _MTPAttentionContext(NamedTuple):
    attention_masks: AttentionMasksType | None
    positions: torch.Tensor | None


class _MTPForwardState(NamedTuple):
    tok_embeddings: Any
    hc_hidden: torch.Tensor
    main_hidden: torch.Tensor


class DeepSeekV4MTPTransformerBlock(TransformerBlock):
    """One DSV4 MTP depth that preserves the multi-stream mHC state."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        moe: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        e_proj: Linear.Config
        h_proj: Linear.Config
        mtp_norm: RMSNorm.Config
        hc_head: HcHead.Config

    def __init__(self, config: Config):
        super().__init__()
        self.moe_enabled = True
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.moe = config.moe.build()
        self.hc_attn_pre = config.hc_attn_pre.build()
        self.hc_ffn_pre = config.hc_ffn_pre.build()
        self.hc_post = config.hc_post.build()
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.e_proj = config.e_proj.build()
        self.h_proj = config.h_proj.build()
        self.mtp_norm = config.mtp_norm.build()
        self.hc_head = config.hc_head.build()

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        mtp_input_embed: torch.Tensor,
        prev_hc_hidden: torch.Tensor,
        mtp_input_ids: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_context: _MTPAttentionContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prev_hc_hidden.ndim != 4:
            raise ValueError(
                "DeepSeek-V4 MTP expects an mHC hidden state with shape "
                "[batch, sequence, hc_mult, hidden], got "
                f"{tuple(prev_hc_hidden.shape)}."
            )

        valid_mask = mtp_input_valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=prev_hc_hidden.dtype)
        prev_hc_hidden = prev_hc_hidden * valid_mask

        hidden = self.e_proj(self.enorm(mtp_input_embed)).unsqueeze(2)
        hidden = hidden + self.h_proj(self.hnorm(prev_hc_hidden))
        residual = hidden
        hidden, post, comb = self.hc_attn_pre(hidden)
        hidden = self.attention(
            self.attention_norm(hidden),
            attention_context.attention_masks,
            attention_context.positions,
        )
        hidden = self.hc_post(hidden, residual, post, comb)

        residual = hidden
        hidden, post, comb = self.hc_ffn_pre(hidden)
        hidden = self.moe(self.ffn_norm(hidden), input_ids=mtp_input_ids)
        next_hc_hidden = self.hc_post(hidden, residual, post, comb)

        prediction_hidden = self.hc_head(next_hc_hidden)
        prediction_hidden = self.mtp_norm(prediction_hidden)
        return next_hc_hidden, prediction_hidden


class DeepSeekV4MTPDecoder(MTPDecoder):
    """DeepSeek-V4 decoder with optional multi-token prediction layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        hc_mult: int = 4
        hc_head: HcHead.Config

        def update_from_config(self, *, config, **kwargs) -> None:
            if not self.mtp_layers:
                Decoder.Config.update_from_config(
                    self,
                    config=config,
                    **kwargs,
                )
                return

            num_main_layers = len(self.layers)
            self.layers.extend(self.mtp_layers)
            try:
                Decoder.Config.update_from_config(
                    self,
                    config=config,
                    **kwargs,
                )
            finally:
                del self.layers[num_main_layers:]

            if config.parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError("MTP does not support pipeline parallelism yet.")

    def __init__(self, config: Config):
        # MTPDecoder builds the community MTP block type. DSV4 owns a
        # different block while retaining the same decoder-level contract.
        Decoder.__init__(self, config)
        self.hc_mult = config.hc_mult
        self.hc_head = config.hc_head.build()

        if not config.mtp_layers:
            self.mtp_layers = None
            return

        self.mtp_layers = ModuleList()
        for layer_config in config.mtp_layers:
            self.mtp_layers.append(layer_config.build())

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        mtp_batch: MTPBatch | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        attention_context = _MTPAttentionContext(attention_masks, positions)
        state = self._forward_main(tokens, attention_context)

        if self.mtp_layers is None:
            if self._skip_lm_head or self.lm_head is None:
                return state.main_hidden
            return self.lm_head(state.main_hidden)

        if state.tok_embeddings is None:
            raise ValueError("DeepSeek-V4 MTP forward requires token embeddings.")

        outputs = [
            state.main_hidden,
            *self._forward_mtp_layers(tokens, state, attention_context, mtp_batch),
        ]
        if not self._skip_lm_head:
            outputs = [self.lm_head(item) if self.lm_head is not None else item for item in outputs]
        return MTPModelOutput(
            outputs,
            mtp_labels=mtp_batch.labels if mtp_batch is not None else None,
        )

    def _forward_main(
        self,
        tokens: torch.Tensor,
        attention_context: _MTPAttentionContext,
    ) -> _MTPForwardState:
        tok_embeddings = self.tok_embeddings
        input_ids = tokens.detach().long()
        hidden = tok_embeddings(tokens) if tok_embeddings is not None else tokens
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for layer in self.layers.values():
            hidden = layer(
                hidden,
                input_ids,
                attention_context.attention_masks,
                attention_context.positions,
            )

        main_hidden = self.hc_head(hidden)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden
        return _MTPForwardState(tok_embeddings, hidden, main_hidden)

    def _forward_mtp_layers(
        self,
        tokens: torch.Tensor,
        state: _MTPForwardState,
        attention_context: _MTPAttentionContext,
        mtp_batch: MTPBatch | None,
    ) -> list[torch.Tensor]:
        if self.mtp_layers is None:
            return []
        if mtp_batch is not None:
            if mtp_batch.input_tokens.shape[-1] != len(self.mtp_layers):
                raise ValueError(
                    "Prepared MTP input depth does not match model layers: "
                    f"{mtp_batch.input_tokens.shape[-1]} inputs for "
                    f"{len(self.mtp_layers)} layers."
                )
            _annotate_mtp_batch(mtp_batch)

        mtp_outputs = []
        prev_hc_hidden = state.hc_hidden
        for depth, layer in enumerate(self.mtp_layers, 1):
            if mtp_batch is None:
                mtp_tokens, valid_mask = roll_mtp_sequence(
                    tokens,
                    shift=depth,
                    fill_value=0,
                    positions=attention_context.positions,
                    return_valid_mask=True,
                )
            else:
                mtp_tokens = mtp_batch.input_tokens[..., depth - 1]
                valid_mask = mtp_batch.input_valid_mask[..., depth - 1]
            prev_hc_hidden, prediction_hidden = layer(
                state.tok_embeddings(mtp_tokens),
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid_mask,
                attention_context,
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs
