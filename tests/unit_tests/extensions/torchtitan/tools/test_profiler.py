# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU tests for the CANN profiler schedule restore logic."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

with patch.dict(
    sys.modules,
    {
        "torch_npu": MagicMock(),
        "torchtitan_npu.patches": MagicMock(),
    },
):
    from torchtitan_npu.config import ProfilerExtensionConfig
    from torchtitan_npu.extensions.torchtitan.tools import profiler as profiler_module
    from torchtitan_npu.extensions.torchtitan.tools.profiler import (
        CANNProfiler,
        _profile_window_schedule,
        _resume_aware_schedule,
    )


pytestmark = pytest.mark.cpu


def test_cann_profiler_config_builds_extension_directly():
    profiler = CANNProfiler.Config().build()

    assert isinstance(profiler, CANNProfiler)


def test_profile_window_schedule_expands_absolute_steps():
    schedule = _profile_window_schedule(
        CANNProfiler.Config(
            extension=ProfilerExtensionConfig(
                profiler_start=5,
                profiler_end=8,
            ),
            profiler_warmup=3,
        )
    )

    assert schedule == {
        "profile_freq": 6,
        "profiler_warmup": 3,
        "profiler_active": 3,
        "profiler_repeat": 1,
        "profiler_skip_first": 1,
        "profiler_skip_first_wait": None,
    }


def test_cann_profiler_applies_absolute_window_schedule(tmp_path):
    schedule_fn = MagicMock()
    torch_profiler = MagicMock()

    with (
        patch.object(
            profiler_module.torch_npu.profiler,
            "schedule",
            return_value=schedule_fn,
        ) as build_schedule,
        patch.object(
            profiler_module.torch_npu.profiler,
            "profile",
            return_value=torch_profiler,
        ),
    ):
        profiler = CANNProfiler.Config(
            enable_profiling=True,
            extension=ProfilerExtensionConfig(
                profiler_start=5,
                profiler_end=8,
            ),
            profiler_warmup=3,
        ).build()
        result = profiler.build_torch_profiler(
            global_step=0,
            base_folder=str(tmp_path),
            leaf_folder="",
        )

    assert result is torch_profiler
    build_schedule.assert_called_once_with(
        wait=0,
        warmup=3,
        active=3,
        repeat=1,
        skip_first=1,
    )


def test_resume_aware_schedule_restores_absolute_position():
    schedule = torch.profiler.schedule(
        wait=0,
        warmup=3,
        active=3,
        repeat=1,
        skip_first=1,
    )
    resumed_schedule = _resume_aware_schedule(schedule, global_step=4)

    # The first action after restoring step 4 is the action for training step 5
    # (schedule position 4), followed by positions 5 and 6.
    assert resumed_schedule(0) == schedule(4)
    assert resumed_schedule(5) == schedule(5)
    assert resumed_schedule(6) == schedule(6)
    assert resumed_schedule(7) == schedule(7)


def test_resume_aware_schedule_keeps_fresh_run_unchanged():
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)

    assert _resume_aware_schedule(schedule, global_step=0) is schedule
