# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: collect training traces with ``torch_npu.profiler``."""

import os
from dataclasses import dataclass

import torch_npu
from torchtitan.config import derive, override
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class NPUProfiler(Profiler):
    @dataclass(kw_only=True, slots=True)
    class Config(Profiler.Config):
        pass

    def build_torch_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        cfg = self._config
        if not cfg.enable_profiling:
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
        assert wait >= 0, (
            "profile_freq must be greater than or equal to warmup + active"
        )

        def _env_bool(name: str) -> bool:
            return os.environ.get(name, "0") == "1"

        profile_with_memory = _env_bool("TORCHTITAN_NPU_PROFILE_WITH_MEMORY")
        profile_with_stack = _env_bool("TORCHTITAN_NPU_PROFILE_WITH_STACK")
        enable_online_parse = _env_bool("TORCHTITAN_NPU_ENABLE_ONLINE_PARSE")

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
    description="NPU profiler using torch_npu.profiler with Ascend-specific options",
)
def npu_profiler_override(cfg: Profiler.Config) -> NPUProfiler.Config:
    return derive(cfg, NPUProfiler.Config)
