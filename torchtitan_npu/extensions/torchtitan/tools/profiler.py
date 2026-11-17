# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect training traces with ``torch_npu.profiler``."""

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import torch
import torch_npu
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler

from torchtitan_npu.config.configs import ProfilerConfig


def _resume_aware_schedule(
    schedule: Callable[[int], object],
    global_step: int,
) -> Callable[[int], object]:
    """Restore a schedule's absolute position when resuming training.

    ``torch.profiler.profile`` evaluates the schedule at step ``0`` while it
    is being constructed.  The trainer then calls ``step()`` with the
    restored global step as the profiler's counter.  Without adapting the
    first lookup, a resumed profiler starts from the schedule's initial state
    and the first training step after resume is recorded with the wrong
    action.

    The initial lookup represents the next training step, whose schedule
    position is ``global_step``.  Subsequent lookups receive the absolute
    profiler step directly.
    """

    if global_step == 0:
        return schedule

    def schedule_fn(step: int) -> object:
        return schedule(global_step if step == 0 else step)

    return schedule_fn


def _profile_window_schedule(cfg: ProfilerConfig) -> dict[str, int | None]:
    """Translate an absolute profiler window into native schedule fields."""

    profile_start = cfg.extension.profiler_start
    profile_end = cfg.extension.profiler_end
    schedule: dict[str, int | None] = {
        "profile_freq": cfg.profile_freq,
        "profiler_warmup": cfg.profiler_warmup,
        "profiler_active": cfg.profiler_active,
        "profiler_repeat": cfg.profiler_repeat,
        "profiler_skip_first": cfg.profiler_skip_first,
        "profiler_skip_first_wait": cfg.profiler_skip_first_wait,
    }
    if profile_start is None and profile_end is None:
        return schedule
    if profile_start is None or profile_end is None:
        raise ValueError("profiler_start and profiler_end must be provided together")
    if profile_start < 1 or profile_end < 1:
        raise ValueError("profiler_start and profiler_end must be positive integers")
    if profile_end <= profile_start:
        raise ValueError("profiler_end must be greater than profiler_start")

    profile_warmup = cfg.profiler_warmup
    if profile_warmup < 0:
        raise ValueError("profiler_warmup must be a non-negative integer")

    profile_skip_first = max(profile_start - profile_warmup - 1, 0)
    profile_warmup_steps = profile_start - 1 - profile_skip_first
    profile_active = profile_end - profile_start
    schedule.update(
        profile_freq=profile_warmup_steps + profile_active,
        profiler_warmup=profile_warmup_steps,
        profiler_active=profile_active,
        profiler_repeat=1,
        profiler_skip_first=profile_skip_first,
    )
    return schedule


class CANNProfiler(Profiler):
    @dataclass(kw_only=True, slots=True)
    class Config(ProfilerConfig):
        """Profiler configuration with the NPU extension namespace."""

    def __init__(
        self,
        config: Config,
        *,
        global_step: int = 0,
        base_folder: str = "",
        leaf_folder: str = "",
    ) -> None:
        config = replace(config, **_profile_window_schedule(config))
        super().__init__(
            config,
            global_step=global_step,
            base_folder=base_folder,
            leaf_folder=leaf_folder,
        )

    def build_torch_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        cfg = cast("CANNProfiler.Config", self._config)
        if not cfg.enable_profiling:
            return None

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if -1 not in cfg.extension.profile_ranks and rank not in cfg.extension.profile_ranks:
            logger.info(
                "Profiling disabled for rank %d; configured profile_ranks=%s",
                rank,
                cfg.extension.profile_ranks,
            )
            return None

        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
        profile_start = cfg.extension.profiler_start
        profile_end = cfg.extension.profiler_end
        if profile_start is not None and profile_end is not None and global_step >= profile_end - 1:
            logger.info(
                "Profiler window [%d, %d) has already passed at restored step %d",
                profile_start,
                profile_end,
                global_step,
            )
            return None

        profile_freq, warmup, active = (
            cfg.profile_freq,
            cfg.profiler_warmup,
            cfg.profiler_active,
        )
        additional_params = {
            key: val
            for key, val in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if val is not None
        }
        wait = profile_freq - (active + warmup)
        if wait < 0:
            raise ValueError("profile_freq must be greater than or equal to warmup + active")

        profile_with_memory = cfg.extension.profile_with_memory
        profile_with_stack = cfg.extension.profile_with_stack
        enable_online_parse = cfg.extension.enable_online_parse

        # NPU profiling accepts only its TensorBoard handler or ``None``.
        if enable_online_parse:
            on_trace_ready = torch_npu.profiler.tensorboard_trace_handler(trace_dir)
        else:
            os.environ["ASCEND_WORK_PATH"] = trace_dir
            on_trace_ready = None

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
        )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        profile_schedule = torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            **additional_params,
        )
        profile_schedule = _resume_aware_schedule(profile_schedule, global_step)

        torch_profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=profile_schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=profile_with_memory,
            with_stack=profile_with_stack,
            experimental_config=experimental_config,
        )
        torch_profiler.step_num = global_step
        torch_profiler.__enter__()
        return torch_profiler
