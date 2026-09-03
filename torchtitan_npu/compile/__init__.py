# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time extensions for NPU models."""

__all__ = ["PatternReplacement", "configure_npu_backend", "register_pre_aot_patterns"]

import importlib
import logging
import os

from .pattern_replacement import PatternReplacement, register_pre_aot_patterns

logger = logging.getLogger(__name__)

for module_path in os.environ.get("TORCHTITAN_NPU_PATTERN_IMPORTS", "").split(","):
    if module_path := module_path.strip():
        importlib.import_module(module_path)


def configure_npu_backend() -> None:
    """Default the Inductor NPU codegen to the AscendC backend.

    torch_npu resolves the Inductor NPU backend per compile as
    ``compile options > torch._inductor.config.npu_backend > TORCHINDUCTOR_NPU_BACKEND``.
    Setting the config layer here makes every ``torch.compile(..., backend="inductor")``
    call default to AscendC codegen, including torchtitan's block-level compiles
    that pass no options.

    ``TORCHINDUCTOR_NPU_BACKEND`` remains the escape hatch: any value preset in
    the environment (``default`` included) is honored and the config layer is
    left untouched, so A/B comparisons and forced rollbacks need no code change.
    """
    if os.getenv("TORCHINDUCTOR_NPU_BACKEND"):
        logger.debug(
            "TORCHINDUCTOR_NPU_BACKEND=%s is set; skipping AscendC default",
            os.environ["TORCHINDUCTOR_NPU_BACKEND"],
        )
        return

    from torch._inductor import config as inductor_config

    # The npu_backend key is registered lazily by torch_npu's patched
    # ConfigModule.get_config_copy; the unpatched config raises on setattr of
    # unregistered keys, so touch the registration once first.
    inductor_config.get_config_copy()  # pyrefly: ignore [missing-attribute]
    try:
        inductor_config.npu_backend = "ascendc"  # pyrefly: ignore [missing-attribute]
    except AttributeError:
        # torch_npu build without the config key: keep its own default rather
        # than failing the package import.
        logger.warning("torch._inductor.config has no npu_backend key; keeping the torch_npu default Inductor backend")
    else:
        logger.info("Inductor NPU backend defaulting to AscendC")
