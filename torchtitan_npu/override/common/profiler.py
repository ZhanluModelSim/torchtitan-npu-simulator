# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: collect training traces with ``torch_npu.profiler``."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class CANNProfiler(Profiler):
    @dataclass(kw_only=True, slots=True)
    class Config(Profiler.Config):
        profile_ranks: Sequence[int]
        """Ranks to profile. ``[-1]`` profiles every rank."""

        profile_with_memory: bool
        """Whether to record memory events in the profiler trace."""

        profile_with_stack: bool
        """Whether to record Python/C++ stack information in the trace."""

        enable_online_parse: bool
        """Whether CANN should parse traces online via its trace handler."""

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
        if -1 not in cfg.profile_ranks and rank not in cfg.profile_ranks:
            logger.info(
                "Profiling disabled for rank %d; configured profile_ranks=%s",
                rank,
                cfg.profile_ranks,
            )
            return None

        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
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

        profile_with_memory = cfg.profile_with_memory
        profile_with_stack = cfg.profile_with_stack
        enable_online_parse = cfg.enable_online_parse

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

        torch_profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=wait,
                warmup=warmup,
                active=active,
                **additional_params,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=profile_with_memory,
            with_stack=profile_with_stack,
            experimental_config=experimental_config,
        )
        torch_profiler.__enter__()
        torch_profiler.step_num = global_step
        return torch_profiler


@override(
    target=Profiler.Config,
    description="CANN profiler using torch_npu.profiler with Ascend-specific options",
)
def cann(
    cfg: Profiler.Config,
    *,
    profile_ranks: Sequence[int] = (-1,),
    profile_with_memory: bool = False,
    profile_with_stack: bool = False,
    enable_online_parse: bool = True,
) -> CANNProfiler.Config:
    return derive(
        cfg,
        CANNProfiler.Config,
        profile_ranks=profile_ranks,
        profile_with_memory=profile_with_memory,
        profile_with_stack=profile_with_stack,
        enable_online_parse=enable_online_parse,
    )
