# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430

import functools
import logging

import torchtitan
from torchtitan.distributed.full_dtensor import (
    validate_config as original_validate_config,
)

logger = logging.getLogger(__name__)


@functools.wraps(original_validate_config)
def patched_validate_config(parallel_dims, model):
    from torchtitan.models.common.attention import ScaledDotProductAttention

    if parallel_dims.cp_enabled and any(isinstance(m, ScaledDotProductAttention) for m in model.modules()):
        raise NotImplementedError(
            f"{parallel_dims.spmd_backend} + CP is not supported with "
            "ScaledDotProductAttention. "
            "Use FlexAttention + CP or disable CP."
        )


def apply() -> None:
    logger.info("[PATCH] torchtitan.distributed.full_dtensor.validate_config -> patched_validate_config")
    torchtitan.distributed.full_dtensor.validate_config = patched_validate_config


apply()
