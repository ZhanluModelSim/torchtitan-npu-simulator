import os
from dataclasses import dataclass

import torch
import torch_npu

from torchtitan.config import derive, override
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class NPUProfiler(Profiler):

    @dataclass(kw_only=True, slots=True)
    class Config(Profiler.Config):
        profile_with_stack: bool = False
        profile_with_memory: bool = False
        enable_online_parse: bool = False

    def build_torch_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        cfg: NPUProfiler.Config = self._config
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

        # torch_npu.profiler.profile on_trace_ready only accepts
        # tensorboard_trace_handler or None (custom callbacks are unsupported).
        if cfg.enable_online_parse:
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
                wait=wait, warmup=warmup, active=active, **additional_params,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=cfg.profile_with_memory,
            with_stack=cfg.profile_with_stack,
            experimental_config=experimental_config,
        )
        torch_profiler.__enter__()
        torch_profiler.step_num = global_step
        return torch_profiler


@override(
    target=Profiler.Config,
    description="NPU profiler using torch_npu.profiler with Ascend-specific options",
)
def profiler(
    cfg: Profiler.Config,
    *,
    profile_with_stack: bool = False,
    profile_with_memory: bool = False,
    enable_online_parse: bool = False,
) -> NPUProfiler.Config:
    return derive(
        cfg,
        NPUProfiler.Config,
        profile_with_stack=profile_with_stack,
        profile_with_memory=profile_with_memory,
        enable_online_parse=enable_online_parse,
    )
