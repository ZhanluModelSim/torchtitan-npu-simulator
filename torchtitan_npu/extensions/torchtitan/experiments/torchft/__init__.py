# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in TorchFT integration for Ascend NPU."""

from typing import TYPE_CHECKING, cast

try:
    import torchft  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "torchft":
        raise
    raise ModuleNotFoundError(
        "The TorchFT experiment requires its optional dependencies: pip install -e '.[torchft]'",
        name="torchft",
    ) from error

from torchtitan_npu.patches import torchft as _torchft_patches  # noqa: F401
from torchtitan_npu.patches.torchtitan.experiments import torchft as _torchtitan_torchft_patches  # noqa: F401

# Activate only after the optional dependency and its device patches are ready.
from .trainer import FaultTolerantTrainerEx, FaultTolerantTrainingEngine, FaultTolerantTrainingEngineEx

if TYPE_CHECKING:
    from torchtitan.training_engine import TrainingEngine


def _new_training_engine(
    cls: type[FaultTolerantTrainingEngine], config: "TrainingEngine.Config | None" = None, **kwargs
) -> FaultTolerantTrainingEngine:
    # Allocate only: Python calls __init__ once on the selected instance.
    # Keep explicit subclasses and the upstream class symbol unchanged.
    if cls is FaultTolerantTrainingEngine and isinstance(config, FaultTolerantTrainerEx.Config):
        cls = FaultTolerantTrainingEngineEx
    return cast("FaultTolerantTrainingEngine", object.__new__(cls))


def apply() -> None:
    """Select the NPU engine only for NPU FT trainer configurations.

    Unlike the ordinary Trainer, the FT constructor does not consult engine_cls.
    This opt-in package is the sole owner of its allocation hook.
    """
    FaultTolerantTrainingEngine.__new__ = staticmethod(_new_training_engine)


apply()
