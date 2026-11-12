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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import spmd_types as spmd
import torch
from torchtitan.components.loss import (
    IGNORE_INDEX,
    BaseLoss,
    ChunkedLossWrapper,
)
from torchtitan.config import derive
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v3.mtp import (
    MTPDecoder,
    MTPLoss,
    roll_mtp_sequence,
)
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model as TorchTitanV4Model
from torchtitan.models.deepseek_v4.mtp import MTPBlock

from torchtitan_npu.models.common.attention_metadata import AttentionMetadataMap, GeometryAttentionMetadataMixin
from torchtitan_npu.models.common.linear import SharedLinear

from .mhc import HcHead  # noqa: TC001


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
    padding_mask: torch.Tensor | None = None,
) -> MTPBatch:
    """Build DeepSeek-V4 MTP inputs before context-parallel sharding.

    ``padding_mask`` is structural padding, rather than the loss mask encoded
    by ``IGNORE_INDEX``.  It must be applied while rolling both the input and
    target streams so a shifted token can never read from a padded tail.  The
    batch is built on the global row before CP sharding; doing this after the
    shard would make a CP boundary look like a document boundary.
    """
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
            padding_mask=padding_mask,
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
                    padding_mask=padding_mask,
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
    token_layout = spmd.SpmdType(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R}, partition_spec=spmd.PartitionSpec(("dp", "cp"), None)
    )
    label_layout = spmd.SpmdType(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.I}, partition_spec=spmd.PartitionSpec(("dp", "cp"), None)
    )
    spmd.assert_type(batch.input_tokens, token_layout)
    spmd.assert_type(batch.input_valid_mask, token_layout)
    spmd.assert_type(batch.labels, label_layout)


def _mtp_labels(
    pred: list[torch.Tensor],
    labels: torch.Tensor,
    positions: torch.Tensor | None,
    padding_mask: torch.Tensor | None = None,
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
                padding_mask=padding_mask,
                fill_value=IGNORE_INDEX,
            ),
        )
        for depth in range(1, num_mtp_layers + 1)
    ]


def _mtp_head_context(lm_head, *, isolate_gradients: bool):
    if not isolate_gradients:
        return nullcontext()
    if not isinstance(lm_head, SharedLinear):
        raise ValueError(
            "mtp_isolate_gradients requires a SharedLinear LM head; "
            f"got {type(lm_head).__name__}. Enable isolation on both the model and loss configurations."
        )
    return lm_head.detached_weight()


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
        mtp_isolate_gradients: bool = False

    def __init__(self, config: Config, *, compile_config=None):
        super().__init__(config)
        self.mtp_scale = config.mtp_scale
        self.mtp_isolate_gradients = config.mtp_isolate_gradients
        self._scaled_loss = _ScaledLoss(self.loss_fn)
        self.loss_fn = self._scaled_loss  # pyrefly: ignore [bad-assignment]

    def __call__(
        self,
        pred: tuple[torch.Tensor, ...],
        labels: tuple[torch.Tensor, ...],
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(pred, tuple) or not isinstance(labels, tuple) or len(pred) != len(labels) or len(pred) < 2:
            raise ValueError("MTP chunked loss requires aligned main and MTP tuples.")
        if global_loss_token_counts is not None and (
            global_loss_token_counts.ndim != 1 or global_loss_token_counts.numel() != len(pred)
        ):
            raise ValueError("MTP chunked loss requires one denominator per objective.")
        counts = (
            (None,) * len(pred)
            if global_loss_token_counts is None
            else tuple(count.clamp_min(1) for count in global_loss_token_counts.unbind())
        )
        mtp_head_context = _mtp_head_context(self.lm_head, isolate_gradients=self.mtp_isolate_gradients)
        self._scaled_loss.scale = 1.0
        total_loss, metrics = super().__call__(pred[0], labels[0], counts[0], **loss_inputs)
        with mtp_head_context:
            for hidden, depth_labels, count in zip(pred[1:], labels[1:], counts[1:], strict=True):
                self._scaled_loss.scale = self.mtp_scale / (len(pred) - 1)
                depth_loss, depth_metrics = super().__call__(hidden, depth_labels, count, **loss_inputs)
                total_loss = total_loss + depth_loss
                metrics = self._combine_chunk_metrics(metrics, depth_metrics)
        return total_loss, metrics


class _MTPAttentionContext(NamedTuple):
    """What every DSV4 layer reads for its attention, main and MTP depths.

    The per-microbatch metadata map (upstream #4897) is built once for the whole
    stack -- main layers and MTP depths together -- and each layer reads the entry
    its own attention backend named.  There is no single-object fallback: the V4
    cores all build their own metadata, so a layer whose key is missing is a
    config that built none for a backend it runs, not a reason to hand it another
    geometry's grid.
    """

    attention_metadata: AttentionMetadataMap
    positions: torch.Tensor | None

    def for_layer(self, layer: Any) -> Any:
        """The metadata one layer reads: its own map entry, keyed by its backend."""
        key = getattr(layer.attention, "attention_metadata_key", type(layer.attention.inner_attention))
        metadata = self.attention_metadata.get(key)
        if metadata is None:
            raise ValueError(f"no attention metadata was built for {key!r}")
        return metadata


class _MTPForwardState(NamedTuple):
    tok_embeddings: Any
    hc_hidden: torch.Tensor
    main_hidden: torch.Tensor


class DeepSeekV4MTPTransformerBlock(MTPBlock):
    """Adapt geometry-keyed metadata to the upstream MTP block interface."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPBlock.Config):
        pass

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        mtp_input_embed,
        prev_hc_hidden,
        mtp_input_ids,
        mtp_input_valid_mask,
        attention_context,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        return super().forward(
            mtp_input_embed,
            prev_hc_hidden,
            mtp_input_ids,
            mtp_input_valid_mask,
            attention_context.for_layer(self),
            attention_context.positions,
            padding_mask=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )


class DeepSeekV4MTPDecoder(GeometryAttentionMetadataMixin, TorchTitanV4Model):
    """DeepSeek-V4 decoder with optional multi-token prediction layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanV4Model.Config):
        hc_mult: int = 4
        hc_head: HcHead.Config
        mtp_isolate_gradients: bool = False

    def __init__(self, config: Config):
        if config.mtp_isolate_gradients:
            if not config.mtp_layers or type(config.lm_head) is not Linear.Config:
                raise ValueError("MTP gradient isolation requires MTP layers and an unmodified Linear LM head.")
            config = derive(config, type(config), lm_head=derive(config.lm_head, SharedLinear.Config))
        super().__init__(config)
        self.mtp_isolate_gradients = config.mtp_isolate_gradients
        if not config.mtp_layers:
            self.mtp_layers = None  # pyrefly: ignore [bad-assignment]

    # Reuse aligned input/label preparation without copying the V3 MTP implementation.
    preprocess_inputs = MTPDecoder.preprocess_inputs

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        *,
        attention_metadata: AttentionMetadataMap,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """The metadata map is the pinned argument's replacement, not a renaming of it.

        The pinned ``MTPDecoder.forward`` takes one mask object for every layer; this decoder
        takes the decoder's map and hands each layer the entry of its own geometry, so the
        parameter is keyword-only and the pinned name would be a lie.
        """
        attention_context = _MTPAttentionContext(attention_metadata, positions)
        if self.mtp_layers is not None:
            if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
                raise ValueError("MTP requires one token tensor per objective.")
            if mtp_input_valid_masks is None or len(mtp_input_valid_masks) != len(self.mtp_layers):
                raise ValueError("MTP requires one validity mask per depth.")
            main_tokens, *depth_tokens = tokens
            mtp_batch = MTPBatch(
                torch.stack(depth_tokens, dim=-1),
                torch.stack(mtp_input_valid_masks, dim=-1),
                torch.empty(0, device=main_tokens.device),
            )
        else:
            if not isinstance(tokens, torch.Tensor):
                raise ValueError("Decoder without MTP requires one token tensor.")
            main_tokens, mtp_batch = tokens, None
        if aux_loss_denominators is not None and aux_loss_denominators.numel() != (
            1 + (0 if self.mtp_layers is None else len(self.mtp_layers))
        ):
            raise ValueError("Expected one routing denominator per objective.")
        state = self._forward_main(
            main_tokens,
            attention_context,
            padding_mask=padding_mask,
            aux_loss_denominator=None if aux_loss_denominators is None else aux_loss_denominators[0],
        )

        if self.mtp_layers is None:
            if self._skip_lm_head or self.lm_head is None:
                return state.main_hidden
            return self.lm_head(state.main_hidden)

        if state.tok_embeddings is None:
            raise ValueError("DeepSeek-V4 MTP forward requires token embeddings.")

        outputs = [
            state.main_hidden,
            *self._forward_mtp_layers(
                main_tokens,
                state,
                attention_context,
                mtp_batch,
                padding_mask=padding_mask,
                aux_loss_denominators=aux_loss_denominators,
            ),
        ]
        if not self._skip_lm_head and self.lm_head is not None:
            outputs[0] = self.lm_head(outputs[0])
            with _mtp_head_context(self.lm_head, isolate_gradients=self.mtp_isolate_gradients):
                outputs[1:] = [self.lm_head(item) for item in outputs[1:]]
        return tuple(outputs)

    def _forward_main(
        self,
        tokens: torch.Tensor,
        attention_context: _MTPAttentionContext,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> _MTPForwardState:
        tok_embeddings = self.tok_embeddings
        input_ids = tokens.detach().long()
        hidden = tok_embeddings(tokens) if tok_embeddings is not None else tokens
        hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for layer in self.layers.values():
            hidden = layer(
                hidden,
                input_ids,
                attention_context.for_layer(layer),
                attention_context.positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
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
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.mtp_layers is None:
            return []
        if mtp_batch is not None and mtp_batch.input_tokens.shape[-1] != len(self.mtp_layers):
            raise ValueError(
                "Prepared MTP input depth does not match model layers: "
                f"{mtp_batch.input_tokens.shape[-1]} inputs for "
                f"{len(self.mtp_layers)} layers."
            )

        mtp_outputs = []
        prev_hc_hidden = state.hc_hidden.detach() if self.mtp_isolate_gradients else state.hc_hidden
        for depth, layer in enumerate(self.mtp_layers, 1):
            if mtp_batch is None:
                mtp_tokens, valid_mask = roll_mtp_sequence(
                    tokens,
                    shift=depth,
                    fill_value=0,
                    positions=attention_context.positions,
                    padding_mask=padding_mask,
                    return_valid_mask=True,
                )
            else:
                mtp_tokens = mtp_batch.input_tokens[..., depth - 1]
                valid_mask = mtp_batch.input_valid_mask[..., depth - 1]
            with torch.no_grad() if self.mtp_isolate_gradients else nullcontext():
                mtp_embed = state.tok_embeddings(mtp_tokens)
            prev_hc_hidden, prediction_hidden = layer(
                mtp_embed,
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid_mask,
                attention_context,
                padding_mask=padding_mask,
                aux_loss_denominator=None if aux_loss_denominators is None else aux_loss_denominators[depth],
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs
