# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os

import torch
import torchtitan.tools.profiling as upstream_profiling
import torchtitan.trainer as trainer_module
from torchtitan.tools.logging import logger

from torchtitan_npu.config.configs import ProfilingConfig


def _find_ascend_pt_dirs(trace_dir: str) -> set[str]:
    profiling_data_dir = os.path.join(trace_dir, "profiling_data")
    if not os.path.isdir(profiling_data_dir):
        return set()

    return {
        os.path.join(profiling_data_dir, name)
        for name in os.listdir(profiling_data_dir)
        if name.endswith("_ascend_pt") and os.path.isdir(os.path.join(profiling_data_dir, name))
    }


def is_profile_enabled(profiling_config: ProfilingConfig) -> bool:
    """Check if profiling is enabled for the current rank."""
    if not profiling_config.enable_profiling:
        return False
    profile_ranks = profiling_config.profile_ranks
    if profile_ranks == [-1]:
        return True
    return torch.distributed.get_rank() in profile_ranks


def _log_profiling_config(rank: int, profiling_config: ProfilingConfig) -> None:
    """Log profiling configuration for the current rank."""
    logger.info(
        f"Profiling enabled for rank {rank} with config: "
        f"profile_step_start={profiling_config.profile_step_start}, "
        f"profile_step_end={profiling_config.profile_step_end}, "
        f"profile_ranks={profiling_config.profile_ranks}, "
        f"profile_record_shapes={profiling_config.profile_record_shapes}, "
        f"profile_with_memory={profiling_config.profile_with_memory}, "
        f"profile_with_stack={profiling_config.profile_with_stack}, "
        f"profile_freq={profiling_config.profile_freq}, "
        f"profiler_active={profiling_config.profiler_active}, "
        f"profiler_warmup={profiling_config.profiler_warmup}"
    )


def _report_offline_profile_data(
    trace_dir: str,
    ascend_pt_dirs_before: set[str],
    rank: int,
) -> None:
    pid = os.getpid()
    pid_marker = f"_{pid}_"
    ascend_pt_dirs = sorted(
        path for path in _find_ascend_pt_dirs(trace_dir) - ascend_pt_dirs_before if pid_marker in os.path.basename(path)
    )
    if ascend_pt_dirs:
        logger.info(
            "Profiling raw data for rank %s (pid %s) saved at %s",
            rank,
            pid,
            ", ".join(ascend_pt_dirs),
        )
        logger.info(
            "For offline parsing, see scripts/parse_profiling_data.py usage in "
            "docs/feature_guides/metrics_and_debugging.md."
        )
    else:
        logger.warning(
            "No new profiling raw data directory found for rank %s (pid %s) under %s",
            rank,
            pid,
            os.path.join(trace_dir, "profiling_data"),
        )


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    if not is_profile_enabled(profiling_config):
        yield None
        return

    # pyrefly: ignore [missing-attribute]
    if not torch.npu.is_available():
        raise RuntimeError("Only NPU is supported currently")
    import torch_npu

    trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
    rank = torch.distributed.get_rank()
    _log_profiling_config(rank, profiling_config)

    profile_step_start = profiling_config.profile_step_start
    profile_step_end = profiling_config.profile_step_end
    if profile_step_start > 0 and profile_step_end > 0:
        active = profile_step_end - profile_step_start
        wait = profile_step_start - global_step - 2
        warmup = 1 if profile_step_start > global_step + 1 else 0
        repeat = 1
    else:
        profile_freq = profiling_config.profile_freq
        warmup = profiling_config.profiler_warmup
        active = profiling_config.profiler_active
        wait = profile_freq - (active + warmup)
        repeat = 0

    if wait < 0:
        logger.warning(
            f"profile_step_start must be greater than {global_step} + 2, "
            f"or profile_freq must be greater or equal to warmup + active, but "
            f"got wait={wait}, profiling will be skipped."
        )
        yield None
        return

    os.makedirs(trace_dir, exist_ok=True)
    enable_online_parse = profiling_config.enable_online_parse
    ascend_pt_dirs_before: set[str] = set()
    if enable_online_parse:
        on_trace_ready_handler = torch_npu.profiler.tensorboard_trace_handler(trace_dir)
        logger.info(f"Profiling active (online parse). Traces will be saved at {trace_dir}")
    else:
        os.environ["ASCEND_WORK_PATH"] = trace_dir
        on_trace_ready_handler = None
        ascend_pt_dirs_before = _find_ascend_pt_dirs(trace_dir)
        logger.info(f"Profiling active (offline mode). Raw data will be saved under {trace_dir}/profiling_data.")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
    )

    try:
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=on_trace_ready_handler,
            record_shapes=profiling_config.profile_record_shapes,
            profile_memory=profiling_config.profile_with_memory,
            with_stack=profiling_config.profile_with_stack,
            experimental_config=experimental_config,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    finally:
        if not enable_online_parse:
            _report_offline_profile_data(trace_dir, ascend_pt_dirs_before, rank)


# Patch both the source module and any module that already bound the symbol
# at import time (Trainer imports `from torchtitan.tools.profiling import
# maybe_enable_profiling`, so its local name is fixed when trainer.py is first
# imported — patching only the source module would not propagate).
upstream_profiling.maybe_enable_profiling = maybe_enable_profiling  # pyrefly: ignore [bad-assignment]
trainer_module.maybe_enable_profiling = maybe_enable_profiling  # pyrefly: ignore [bad-assignment]
