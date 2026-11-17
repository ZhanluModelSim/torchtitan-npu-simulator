# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CV-chunk CANN capture, parsing and per-rank isolation."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def cann_profile() -> Iterator[None]:
    """Capture one replay; callers synchronize within their measured scope."""
    import torch_npu

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            record_op_args=False,
        ),
    ) as profiler:
        yield
        profiler.step()


def parse_cann_profile(work_path: Path) -> Path:
    """Synchronously parse this isolated capture and return its output directory."""
    from torch_npu.profiler.profiler import analyse

    paths = sorted(work_path.rglob("*_ascend_pt"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one CANN capture under {work_path}, found {len(paths)}")
    analyse(str(paths[0]), max_process_number=1)
    output = paths[0] / "ASCEND_PROFILER_OUTPUT"
    if not all((output / name).is_file() for name in ("trace_view.json", "kernel_details.csv")):
        raise RuntimeError(f"Native profiler parsing incomplete: {output}")
    logger.info("Parsed native NPU profile: %s", paths[0])
    return output


@contextmanager
def isolated_cann_profiler_work_path(root: Path, *, prefix: str) -> Iterator[Path]:
    """Retain rank 0 raw/parsed data, including partial captures on failure.

    The caller must stop and synchronously parse the profiler inside this
    context. Restore ASCEND_WORK_PATH so other profiler sessions stay isolated.
    """
    previous = os.environ.get("ASCEND_WORK_PATH")
    rank = int(os.getenv("RANK", "0"))
    # Every rank calibrates locally; only rank 0 retains native artifacts.
    with ExitStack() as stack:
        if rank != 0:
            temporary: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="npu_chunk_profile_")
            work_path = Path(stack.enter_context(temporary))
        else:
            root = root.expanduser().resolve() / prefix / f"rank{rank}"
            root.mkdir(parents=True, exist_ok=True)
            work_path = Path(tempfile.mkdtemp(prefix="session_", dir=root))
        os.environ["ASCEND_WORK_PATH"] = str(work_path)
        try:
            yield work_path
        finally:
            if previous is None:
                os.environ.pop("ASCEND_WORK_PATH", None)
            else:
                os.environ["ASCEND_WORK_PATH"] = previous
