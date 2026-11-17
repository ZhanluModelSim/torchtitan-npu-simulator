# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4116
# Remove this module (and its activation import in
# ``torchtitan_npu/patches/torchtitan/__init__.py``) once the pinned
# ``torchtitan`` dependency carries #4116 natively.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport of upstream's model-owned preprocessing seam onto the validator.

The validator's post-dataloading step is the trainer's step once more (0.3.0
even says so in a TODO); upstream #4116 answers that by giving the model the
seam.  This patch makes the validator call ``Decoder.preprocess_inputs``
instead of duplicating the mask build, the CP shard and the SPMD annotation.

The validator and the trainer therefore share one implementation.  One
consequence worth stating: the shared seam annotates SPMD types on the
``spmd_types`` backend, which the validator's 0.3.0 body did not do (it only
handled ``full_dtensor``) — the trainer did, and upstream's deduplicated
version does.
"""

import functools
import logging

from torchtitan.components.validate import Validator

logger = logging.getLogger(__name__)

original_post_dataloading_process = Validator.post_dataloading_process


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels, model_parts):
    """Delegate the dataloader-to-forward preprocessing to the model (#4116)."""
    model = model_parts[0]
    preprocess_inputs = getattr(model, "preprocess_inputs", None)
    if preprocess_inputs is None:
        return original_post_dataloading_process(self, input_dict, labels, model_parts)
    return preprocess_inputs(
        input_dict,
        labels,
        parallel_dims=self.parallel_dims,
        device=input_dict["input"].device,
        parallelism=self.parallelism,
    )


def apply() -> None:
    logger.info("[PATCH] Validator.post_dataloading_process -> patched_post_dataloading_process")
    Validator.post_dataloading_process = patched_post_dataloading_process


apply()
