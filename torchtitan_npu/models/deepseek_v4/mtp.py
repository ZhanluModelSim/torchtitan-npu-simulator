# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import spmd_types as spmd
import torch
from torchtitan.components.loss import IGNORE_INDEX, BaseLoss, ChunkedLossWrapper
from torchtitan.models.deepseek_v3.mtp import MTPLoss as MTPLoss
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence

from torchtitan_npu.models.common.linear import SharedLinear


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
        super().__init__(config, compile_config=compile_config)
        self.mtp_scale = config.mtp_scale
        self.mtp_isolate_gradients = config.mtp_isolate_gradients
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
        padding_mask = loss_inputs.pop("padding_mask", None)
        mtp_labels = _mtp_labels(pred, labels, positions, padding_mask)
        branch_scale = self.mtp_scale / len(mtp_labels)
        head_context = _mtp_head_context(self.lm_head, isolate_gradients=self.mtp_isolate_gradients)

        self._scaled_loss.scale = 1.0
        total_loss, metrics = super().__call__(
            pred[0],
            labels,
            global_valid_tokens,
            **loss_inputs,
        )
        # Keep the real head visible to the upstream FSDP chunk lifecycle.
        with head_context:
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
