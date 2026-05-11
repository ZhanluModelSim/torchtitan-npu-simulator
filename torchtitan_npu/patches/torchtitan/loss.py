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

import torch
from torchtitan.components import loss as loss_utils
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.tools.logging import logger

from ._trainer_config_stash import get_active_trainer_config


def multi_token_cross_entropy_loss(
    preds: list[torch.Tensor],
    labels: torch.Tensor,
    num_mtp_modules: int,
    mtp_loss_weight: float,
) -> torch.Tensor:
    seq_len = preds[0].shape[1]
    main_loss = cross_entropy_loss(preds[0], labels[:, :seq_len])
    mtp_loss = 0

    for label_offset, pred in enumerate(preds[1:], 1):
        end_idx = label_offset + seq_len
        loss = cross_entropy_loss(pred, labels[:, label_offset:end_idx])
        loss = loss / num_mtp_modules
        mtp_loss = mtp_loss + loss
    return main_loss + mtp_loss * mtp_loss_weight


def mtp_build_cross_entropy_loss(compile_config, **kwargs):
    del kwargs  # delete any unused arguments

    trainer_config = get_active_trainer_config()
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
        loss_fn = cross_entropy_loss

    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


loss_utils.build_cross_entropy_loss = mtp_build_cross_entropy_loss
