# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.components.checkpoint import DATALOADER, LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE

from .cache import IndexStoreDtype  # noqa: TC001 - required by tyro at runtime
from .detector import LossSpikeDetector


@dataclass(kw_only=True, slots=True)
class AnticipatoryRoutingConfig:
    """When and how long to train with stale routing indices."""

    enable: bool = False
    """Enable anticipatory routing without replacing the upstream train loop."""

    delay_steps: int = 16
    """Optimizer steps between computing a batch's routing indices and training
    on that batch. Also the number of warmup steps: entering the mode
    pre-computes indices for this many steps' worth of data before the first
    anticipatory optimizer step. Zero means capture and consume a batch in the
    same step, which reproduces standard training exactly and exists for
    numerical validation.

    The system allocates cached microbatch data and expert IDs as the warmup
    queue fills to delay_steps entries (one optimizer step per entry). Memory
    usage therefore grows with delay_steps; an oversized warmup can exhaust
    available memory and cause OOM: microbatches consume CPU memory (including
    pinned memory when supplied by the loader), while expert IDs consume device memory.
    ACTIVE prefetch temporarily holds one additional step before dequeuing.
    """

    active_steps: int = 500
    """Optimizer steps to spend in anticipatory mode before reverting."""

    max_rollbacks: int = 3
    """Upper bound on rollbacks in one run, so a persistently unstable run
    cannot loop forever."""

    index_store_dtype: IndexStoreDtype = "auto"
    """Integer width the cached indices are stored in. ``auto`` picks the
    narrowest dtype that can hold an expert id."""

    detector: LossSpikeDetector.Config = field(default_factory=LossSpikeDetector.Config)

    def __post_init__(self) -> None:
        if self.delay_steps < 0:
            raise ValueError("delay_steps cannot be negative.")
        if self.active_steps < 1:
            raise ValueError("active_steps must be at least 1.")
        if self.max_rollbacks < 0:
            raise ValueError("max_rollbacks cannot be negative.")


def validate_anticipatory_config(config):
    if not config.anticipatory.enable:
        return
    if config.parallelism.pipeline_parallel_degree > 1:
        raise ValueError("Anticipatory routing requires pipeline_parallel_degree=1")
    if config.debug.moe_force_load_balance:
        raise ValueError("moe_force_load_balance bypasses cached expert selection")
    if config.anticipatory.max_rollbacks == 0:
        return
    if not config.checkpoint.enable or config.checkpoint.load_only:
        raise ValueError("Anticipatory routing requires checkpoint saving to be enabled")
    required = {MODEL, DATALOADER, LR_SCHEDULER, OPTIMIZER, TRAIN_STATE}
    if getattr(getattr(config, "ema_weights", None), "enable", False):
        from torchtitan_npu.patches.torchtitan.components.ema import EMA_OPTIMIZER

        required.add(EMA_OPTIMIZER)
    excluded = required.intersection(config.checkpoint.exclude_from_loading)
    if excluded:
        raise ValueError(f"Cannot exclude recovery states: {sorted(excluded)}")
