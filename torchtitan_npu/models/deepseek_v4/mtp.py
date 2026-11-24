# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__all__ = [
    "MTPChunkedLossWrapper",
    "MTPLoss",
    "roll_mtp_sequence",
]

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import spmd_types as spmd
from torchtitan.components.loss import (
    BaseLoss,
    ChunkedLossWrapper,
)
from torchtitan.models.deepseek_v3.mtp import (
    MTPLoss,
    roll_mtp_sequence,
)

from torchtitan_npu.models.common.linear import SharedLinear

if TYPE_CHECKING:
    import torch


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
        with spmd.no_typecheck():
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
