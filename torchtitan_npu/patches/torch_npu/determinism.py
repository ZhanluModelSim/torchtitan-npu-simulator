# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep NPU deterministic environment in sync with PyTorch's switch.

``--debug.deterministic`` is consumed by TorchTitan through
``torch.use_deterministic_algorithms``. NPU additionally requires
``CLOSE_MATMUL_K_SHIFT=1`` and ``HCCL_DETERMINISTIC=true``. The wrapper sets
those variables when deterministic mode is enabled and removes them when it is
disabled.
"""

import functools
import os

import torch
from torchtitan.tools.logging import logger

_NPU_DETERMINISTIC_ENV = {
    "HCCL_DETERMINISTIC": "true",
    "CLOSE_MATMUL_K_SHIFT": "1",
}

original_use_deterministic_algorithms = torch.use_deterministic_algorithms


@functools.wraps(original_use_deterministic_algorithms)
def patched_use_deterministic_algorithms(mode, *args, **kwargs):
    if mode:
        os.environ.update(_NPU_DETERMINISTIC_ENV)
        logger.info("NPU deterministic env enabled: %s", ", ".join(_NPU_DETERMINISTIC_ENV))
    else:
        for key in _NPU_DETERMINISTIC_ENV:
            os.environ.pop(key, None)
        logger.info("NPU deterministic env disabled")
    return original_use_deterministic_algorithms(mode, *args, **kwargs)


def apply() -> None:
    logger.info("[PATCH] torch.use_deterministic_algorithms -> patched_use_deterministic_algorithms")
    torch.use_deterministic_algorithms = patched_use_deterministic_algorithms


apply()
