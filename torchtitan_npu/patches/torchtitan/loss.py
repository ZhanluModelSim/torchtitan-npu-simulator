# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/components/loss.py

This patch replaces `build_cross_entropy_loss` with an MTP-aware builder
that returns `multi_token_cross_entropy_loss` when the active Trainer.Config
has `training.num_mtp_modules > 0`, otherwise falls back to the upstream
`cross_entropy_loss`.

MTP fields (`num_mtp_modules`, `mtp_loss_weight`) are recovered from the
active `Trainer.Config` via `_trainer_config_stash` (shared with the
hf_datasets patch). Upstream TrainingConfig has no MTP fields, so this
patch is a no-op when the npu-side Training subclass that introduces those
fields is not in use.
"""

import functools
from typing import Any, cast

import torch
from torchtitan.components import loss as loss_utils
from torchtitan.components.loss import IGNORE_INDEX, cross_entropy_loss
from torchtitan.tools.logging import logger

from ._trainer_config_stash import get_trainer_config


def _prepare_labels_for_compact_logits(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.reshape(-1)
    if labels.numel() == preds.shape[0]:
        return labels

    compact_labels = labels[labels.ne(IGNORE_INDEX)]
    if compact_labels.numel() != preds.shape[0]:
        raise RuntimeError(
            "Compact logits loss expects labels to match flattened logits: "
            f"got logits T={preds.shape[0]}, labels={labels.numel()}, "
            f"non-ignored labels={compact_labels.numel()}."
        )
    return compact_labels


def compact_cross_entropy_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if preds.dim() == 2:
        labels = _prepare_labels_for_compact_logits(preds, labels)
        return torch.nn.functional.cross_entropy(
            preds.float(),
            labels,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
    return cross_entropy_loss(preds, labels)


def multi_token_cross_entropy_loss(
    preds: list[torch.Tensor],
    labels: torch.Tensor,
    num_mtp_modules: int,
    mtp_loss_weight: float,
) -> torch.Tensor:
    if preds[0].dim() == 2:
        if labels.dim() != 2 or labels.shape[0] < len(preds):
            raise RuntimeError(
                "Compact logits MTP loss expects labels with shape [num_outputs, T], "
                f"got labels={tuple(labels.shape)} for {len(preds)} prediction tensor(s)."
            )
        main_loss = compact_cross_entropy_loss(preds[0], labels[0])
        mtp_loss = 0
        for label_offset, pred in enumerate(  # pyrefly: ignore [bad-assignment]
            preds[1:], 1
        ):
            loss = compact_cross_entropy_loss(pred, labels[label_offset])
            loss = loss / num_mtp_modules
            mtp_loss = mtp_loss + loss
        return main_loss + mtp_loss * mtp_loss_weight

    seq_len = preds[0].shape[1]
    main_loss = cross_entropy_loss(preds[0], labels[:, :seq_len])
    mtp_loss = 0

    for label_offset, pred in enumerate(  # pyrefly: ignore [bad-assignment]
        preds[1:], 1
    ):
        end_idx = label_offset + seq_len
        loss = cross_entropy_loss(pred, labels[:, label_offset:end_idx])
        loss = loss / num_mtp_modules
        mtp_loss = mtp_loss + loss
    return main_loss + mtp_loss * mtp_loss_weight


def mtp_build_cross_entropy_loss(compile_config, **kwargs):
    del kwargs  # delete any unused arguments

    trainer_config = get_trainer_config()
    num_mtp_modules = 0
    mtp_loss_weight = 0.0
    if trainer_config is not None:
        num_mtp_modules = getattr(trainer_config.training, "num_mtp_modules", 0)
        mtp_loss_weight = getattr(trainer_config.training, "mtp_loss_weight", 0.0)

    if num_mtp_modules > 0:
        loss_fn = functools.partial(
            multi_token_cross_entropy_loss,
            num_mtp_modules=num_mtp_modules,
            mtp_loss_weight=mtp_loss_weight,
        )
        logger.info("Applying loss = main_loss + mtp_loss to the model")
    else:
        loss_fn = compact_cross_entropy_loss

    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


loss_utils.build_cross_entropy_loss = cast("Any", mtp_build_cross_entropy_loss)
