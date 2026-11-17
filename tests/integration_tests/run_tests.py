#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/2807d3f550fe27db18bd9395ba63176364eaed6d/tests/integration_tests/run_tests.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# torchtitan-npu override: the runner consumes the case definition at runtime.
from tests.integration_tests import OverrideDefinitions  # noqa: TC001
from tests.integration_tests.deepseek_v3_2 import build_deepseek_v3_2_test_list
from tests.integration_tests.deepseek_v4 import (
    build_deepseek_v4_checkpoint_resume_test_list,
    build_deepseek_v4_quantized_test_list,
    build_deepseek_v4_test_list,
)
from tests.integration_tests.deepseek_v4_1 import (
    build_deepseek_v4_1_checkpoint_test_list,
    build_deepseek_v4_1_fused_test_list,
    build_deepseek_v4_1_golden_test_list,
    build_deepseek_v4_1_quantized_test_list,
    build_deepseek_v4_1_test_list,
)
from tests.integration_tests.ema import assert_ema_checkpoint_written, build_ema_test_list
from tests.integration_tests.loss_compare import (
    assert_losses_equal,
    compare_checkpoint_metrics,
    extract_losses_from_tensorboard,
    log_print,
    read_losses_from_file,
)
from tests.integration_tests.qwen3_5 import build_qwen3_5_test_list


def build_models_test_list() -> list[OverrideDefinitions]:
    """Return the model integration cases for the default smoke suite."""

    # The DeepSeek-V4.1 golden case is back in the default suite alongside the
    # V4.1 LoRA cases (build_deepseek_v4_1_test_list). The V4 checkpoint-resume
    # and EMA groups stay out: run in the A3 CI pool they fail inside the HCCL
    # scatter AI-CPU kernel during `dcp.save`
    # (soName=libscatter_aicpu_kernel.so, funcName=HcclLaunchAicpuKernelA3 ->
    # "ACL stream synchronize failed, error code:507018"), which is why !812
    # pulled them out. Both remain available through the dedicated
    # `deepseek_v4_checkpoint` and `ema` suites until the runtime supports that
    # checkpoint path.
    return (
        build_deepseek_v4_test_list()
        + build_deepseek_v4_1_test_list()
        + build_deepseek_v4_1_golden_test_list()
        + build_deepseek_v3_2_test_list()
        + build_qwen3_5_test_list()
    )


def build_engram_hf_test_list():
    # Import the hardware-dependent validation entry only for this opt-in suite.
    from tests.integration_tests.engram_hf import build_engram_hf_test_list as build

    return build()


def build_engram_cp_test_list():
    from tests.integration_tests.engram_cp import build_engram_cp_test_list as build

    return build()


# torchtitan-npu override: register the DeepSeek-V4, V4.1 and V3.2 NPU suites.
_TEST_SUITES_FUNCTION = {
    "models": build_models_test_list,
    "deepseek_v3_2": build_deepseek_v3_2_test_list,
    "deepseek_v4": build_deepseek_v4_test_list,
    "deepseek_v4_1": build_deepseek_v4_1_test_list,
    "deepseek_v4_1_checkpoint": build_deepseek_v4_1_checkpoint_test_list,
    "deepseek_v4_1_fused": build_deepseek_v4_1_fused_test_list,
    "deepseek_v4_1_golden": build_deepseek_v4_1_golden_test_list,
    "deepseek_v4_1_quantized": build_deepseek_v4_1_quantized_test_list,
    "deepseek_v4_checkpoint": build_deepseek_v4_checkpoint_resume_test_list,
    "deepseek_v4_quantized": build_deepseek_v4_quantized_test_list,
    "ema": build_ema_test_list,
    "deepseek_v4_1_engram_hf": build_engram_hf_test_list,
    "deepseek_v4_1_engram_cp": build_engram_cp_test_list,
    "qwen3_5": build_qwen3_5_test_list,
}

# Held while a test writes its captured output so concurrent tests do not
# interleave their lines.
_OUTPUT_LOCK = threading.Lock()


@dataclass
class _CaseTiming:
    """Wall time of one integration case, for the end-of-run summary."""

    test_name: str
    ngpu: int
    start: float
    elapsed: float


# torchtitan-npu override: every case records its wall time here so the runner
# can print the per-case cost once the run is over. Cases run concurrently, so
# the list has its own lock.
_CASE_TIMINGS: list[_CaseTiming] = []
_CASE_TIMINGS_LOCK = threading.Lock()


class GPUPool:
    """Allocator for a fixed set of physical NPU ids.

    ``acquire(n)`` blocks until ``n`` NPUs are free and returns a sorted list
    of ids; ``release`` returns them to the pool.

    torchtitan-npu override: the pool records an allocation timeline so CI
    canary runs can report scheduling efficiency directly (busy histogram,
    NPU utilization, allocation size counts, and case-overlap savings)
    instead of reconstructing overlaps from per-case timestamps by hand.
    """

    def __init__(self, ids: list[int]):
        self._free: list[int] = sorted(ids)
        self._cond = threading.Condition()
        self.total = len(self._free)
        self._alloc_counts: dict[int, int] = {}
        # (segment_start, segment_end, npus_in_use, case_holders); the open
        # segment is kept as _segment_start and closed on the next change.
        self._segments: list[tuple[float, float, int, int]] = []
        self._segment_start: float | None = None
        self._in_use = 0
        self._holders = 0

    def _record_state_locked(self, holders_delta: int = 0) -> None:
        """Close the open segment at its pre-change level, reopen at current."""
        now = time.monotonic()
        if self._segment_start is not None:
            self._segments.append((self._segment_start, now, self._in_use, self._holders))
        self._holders += holders_delta
        self._in_use = self.total - len(self._free)
        self._segment_start = now

    def acquire(self, n: int) -> list[int]:
        with self._cond:
            while len(self._free) < n:
                self._cond.wait()
            chosen = sorted(self._free[:n])
            self._free = self._free[n:]
            self._alloc_counts[n] = self._alloc_counts.get(n, 0) + 1
            self._record_state_locked(+1)
            return chosen

    def release(self, gpus: list[int]) -> None:
        with self._cond:
            self._free.extend(gpus)
            self._record_state_locked(-1)
            self._cond.notify_all()

    def stats(self) -> list[str]:
        """Summarize the allocation timeline, or ``[]`` if nothing ran.

        Two lines: NPU packing (window, utilization, busy histogram,
        allocation counts) and case overlap (sequential total as the
        serial-equivalent time vs the wall window, time saved, and how long
        1/2/3+ cases ran concurrently). The window spans the first
        ``acquire`` to the last ``release`` of a non-empty allocation, so
        time spent idle before the first case or after the last one does
        not dilute the ratios, while idle gaps *between* cases do count
        against them.
        """
        with self._cond:
            segments = list(self._segments)
            if self._segment_start is not None:
                segments.append((self._segment_start, time.monotonic(), self._in_use, self._holders))
                self._segments, self._segment_start = segments, None
        active = [seg for seg in segments if seg[2] > 0]
        if not active:
            return []
        window = active[-1][1] - active[0][0]
        busy: dict[int, float] = {}
        conc: dict[int, float] = {}
        npu_seconds = 0.0
        sequential_seconds = 0.0
        for _start, end, level, holders in segments:
            duration = end - _start
            busy[level] = busy.get(level, 0.0) + duration
            conc[holders] = conc.get(holders, 0.0) + duration
            npu_seconds += duration * level
            sequential_seconds += duration * holders
        histogram = ", ".join(f"{level}/{self.total} busy {busy[level]:.1f}s" for level in sorted(busy, reverse=True))
        allocations = ", ".join(f"{count}x({n} NPU)" for n, count in sorted(self._alloc_counts.items(), reverse=True))
        capacity = self.total * window
        pool_line = (
            f"pool: window {window:.1f}s, utilization {100 * npu_seconds / capacity:.0f}% "
            f"({npu_seconds:.1f} of {capacity:.1f} NPU-s); "
            f"busy histogram [{histogram}]; allocations [{allocations}]"
        )
        # Sequential time = integral of the concurrent-case count, i.e. the
        # wall time the cases would need one after another (each case also
        # covers its loss validation); the saved time is what parallel
        # packing removed.
        saved = sequential_seconds - window
        overlap = sum(v for k, v in conc.items() if k >= 2)
        conc_hist = ", ".join(f"{k}x {conc[k]:.1f}s" for k in sorted(conc, reverse=True))
        overlap_line = (
            f"overlap: sequential {sequential_seconds:.1f}s vs window {window:.1f}s -> "
            f"saved {saved:.1f}s ({100 * saved / sequential_seconds:.0f}%); concurrency "
            f"[{conc_hist}], 2+ concurrent {overlap:.1f}s ({100 * overlap / window:.0f}%)"
        )
        return [pool_line, overlap_line]


def _detect_visible_npu_ids() -> list[int]:
    """Ask the CANN runtime which NPUs this environment can actually use.

    With ``ASCEND_RT_VISIBLE_DEVICES`` unset the runtime exposes physical ids
    ``0..device_count-1`` — the same numbering child processes see — so the
    count is a faithful enumeration of usable ids.
    """
    try:
        import torch
        import torch_npu  # noqa: F401  # registers torch.npu
    except ImportError as exc:
        raise RuntimeError(
            "Cannot enumerate the visible NPUs because torch_npu is not "
            f"importable in the runner environment ({exc}). Export "
            "ASCEND_RT_VISIBLE_DEVICES to declare the allocation explicitly."
        ) from exc
    try:
        count = torch.npu.device_count()
    except Exception as exc:  # driver/runtime problems must not deadlock acquire()
        raise RuntimeError(f"torch.npu.device_count() failed: {exc}") from exc
    if not count:
        raise RuntimeError(
            "No NPU is visible in the runner environment "
            "(torch.npu.device_count() == 0); cannot build the parallel pool."
        )
    return list(range(count))


def _resolve_npu_pool_ids(requested: int) -> list[int]:
    """Resolve the physical NPU ids the pool may hand out.

    Precedence:

    1. ``ASCEND_RT_VISIBLE_DEVICES`` inherited by the runner process — CI
       allocates each job a subset of the machine's NPUs this way, so the
       pool must use exactly those ids: anything else would point children
       at NPUs owned by other jobs (or at ids that do not exist). A
       ``--ngpu`` larger than that explicit allocation is a hard error.
    2. Otherwise, the ids the CANN runtime actually exposes. ``requested``
       is an upper bound here, not a device census, and is capped to the
       real devices with a warning.

    torchtitan-npu override: upstream torchtitan builds its pool from
    ``range(args.ngpu)`` because its CI GPUs are enumerated 0..n-1. That
    fabrication broke the first parallel CI canary here: the runner exposed
    4 NPUs while ``--ngpu 8`` was passed, so every case pinned to a
    nonexistent id died at NPU init (CANN ``GetVisibleDevices: ... input
    data range[0-4)``), torchtitan's device probe fell back to "cuda", and
    the CPU-only CI torch aborted with ``module 'torch._C' has no attribute
    '_cuda_setDevice'``. The pool must come from real visibility instead.
    """
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if inherited:
        try:
            visible = sorted({int(part) for part in inherited.split(",") if part.strip()})
        except ValueError as exc:
            raise RuntimeError(f"Cannot parse ASCEND_RT_VISIBLE_DEVICES={inherited!r}: {exc}") from exc
        if not visible:
            raise RuntimeError(f"ASCEND_RT_VISIBLE_DEVICES={inherited!r} lists no NPU ids")
        if requested > len(visible):
            raise RuntimeError(
                f"--ngpu={requested} exceeds the {len(visible)} NPU(s) visible via "
                f"ASCEND_RT_VISIBLE_DEVICES={inherited!r}"
            )
        return visible[:requested]
    visible = _detect_visible_npu_ids()
    if requested > len(visible):
        log_print(
            f"--ngpu={requested} exceeds the {len(visible)} NPU(s) the runtime exposes; "
            "capping the pool to the real devices"
        )
    return visible[:requested]


def _terminate_process_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Terminate ``proc``'s whole process group, escalating TERM to KILL.

    torchtitan-npu override: each child runs in its own session
    (``start_new_session=True``), so the child's pid doubles as its pgid and
    killing the group reaps ``bash scripts/run_train.sh`` -> ``torchrun`` and
    every rank worker it spawned. Killing only the direct child — what a plain
    ``subprocess.run(timeout=...)`` does — would orphan the rank processes and
    leave NPU devices occupied for the rest of the CI run.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _run_cmd(
    cmd: list[str],
    env: dict[str, str],
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` (an argv list, no shell), capturing merged stdout/stderr.

    Output is *not* streamed to the parent in real time: when running tests
    concurrently we want each test's log to appear as one contiguous block
    rather than interleaved line-by-line with other tests.

    On timeout the child's whole process group is terminated (see
    ``_terminate_process_group``) and a synthetic ``CompletedProcess`` with
    ``returncode=-1`` is returned, with ``stdout`` populated with whatever the
    child had emitted so far, so callers do not need to special-case
    ``TimeoutExpired``.
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout or "", stderr=None)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        stdout, _ = proc.communicate()
        return subprocess.CompletedProcess(cmd, -1, stdout=stdout or "", stderr=None)


def _emit_block(prefix: str, header: str, body: str, footer: str = "") -> None:
    """Atomically write a multi-line block prefixed with ``[prefix] ``.

    Holds the global output lock for the entire block so concurrent tests do
    not interleave their lines.
    """
    with _OUTPUT_LOCK:
        sys.stderr.write(header)
        for line in body.splitlines():
            sys.stderr.write(f"[{prefix}] {line}\n")
        if footer:
            sys.stderr.write(footer)
        sys.stderr.flush()


# torchtitan-npu override: reference losses are selected
# from the repository using the case name.
_GOLDEN_DIR = Path(__file__).parents[1] / "assets" / "losses"
# Common launch options. Loss-checking cases additionally enable deterministic
# execution so the produced trajectory can be compared exactly.
DEFAULT_TRAIN_ARGS = (
    "--metrics.enable_tensorboard",
    "--metrics.log_freq=1",
    "--metrics.save_tb_folder=tb",
    "--dataloader.dataset-path=tests/assets/c4_test",
    "--training.disable-cuda-graphs",
)
DETERMINISTIC_ARGS = (
    "--debug.deterministic",
    "--debug.seed=42",
)


def _golden_file_for(test_flavor: OverrideDefinitions) -> Path | None:
    if not test_flavor.check_loss:
        return None
    return _GOLDEN_DIR / f"{test_flavor.test_name}.txt"


def _check_phase_results(
    test_flavor: OverrideDefinitions,
    test_name: str,
    case_dir: Path,
    idx: int,
    golden_losses: dict[int, float] | None,
) -> None:
    """Validate one phase's TensorBoard trajectory (steps and golden losses)."""

    if test_flavor.expected_steps is None and not test_flavor.check_loss:
        return
    tb_folder = f"tb_phase_{idx}" if test_flavor.expected_steps is not None else "tb"
    test_losses = extract_losses_from_tensorboard(case_dir / "test_run", tb_folder)
    if test_flavor.expected_steps is not None:
        expected_steps = set(test_flavor.expected_steps[idx])
        if set(test_losses) != expected_steps:
            raise RuntimeError(
                f"{test_name} phase {idx}: expected steps {sorted(expected_steps)}, got {sorted(test_losses)}"
            )
    if test_flavor.check_loss:
        assert golden_losses is not None
        # torchtitan-npu override: on mismatch, print losses in the
        # reference-file format to simplify deliberate regeneration.
        try:
            assert_losses_equal(golden_losses, test_losses)
        except AssertionError:
            mismatch_body = "\n".join(f"[GOLDEN_MISMATCH] {step} {test_losses[step]}" for step in sorted(test_losses))
            _emit_block(
                test_name,
                f"[GOLDEN_MISMATCH] {test_name} — dumping actual losses for regeneration:\n",
                mismatch_body,
            )
            raise


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    module: str | None = None,
    config: str | None = None,
    *,
    golden_file: str | Path | None = None,
    # ``gpu_ids`` is set only in parallel mode; sequential runs leave the
    # child process to use all visible NPUs.
    gpu_ids: list[int] | None = None,
) -> None:
    """Run one case, optionally validating its exact loss trajectory."""

    test_name = test_flavor.test_name
    case_dir = Path(output_dir) / test_name
    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    if test_flavor.expected_steps is not None and len(test_flavor.expected_steps) != len(test_flavor.override_args):
        raise ValueError(f"Expected one step sequence per phase for {test_name}")
    if test_flavor.check_resume and (test_flavor.expected_steps is None or len(test_flavor.override_args) != 2):
        raise ValueError(f"Resume comparison requires two phases with expected steps for {test_name}")

    # When running in parallel, pin each test to a disjoint subset of physical
    # NPUs. torchtitan-npu override: ASCEND_RT_VISIBLE_DEVICES is the Ascend
    # equivalent of upstream's CUDA_/HIP_VISIBLE_DEVICES pinning, applied
    # through the child environment rather than a shell prefix.
    #
    # Assignment order preserves the historical precedence of the old shell
    # command line: per-case ``env_vars`` override --module/--config, while
    # NGPU/LOG_RANK always come last.
    env = os.environ.copy()
    if gpu_ids is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    if module is not None:
        env["MODULE"] = module
    if config is not None:
        env["CONFIG"] = config
    if test_flavor.env_vars:
        env.update(test_flavor.env_vars)
    env["NGPU"] = str(test_flavor.ngpu)
    env["LOG_RANK"] = all_ranks

    golden_losses = None
    if test_flavor.check_loss:
        if golden_file is None:
            raise ValueError(f"golden_file is required when check_loss=True for {test_name}")
        golden_losses = read_losses_from_file(golden_file)

    for idx, override_arg in enumerate(test_flavor.override_args):
        # torchtitan-npu override: launch the case-specific training script
        # directly with an argv list and pass settings through the environment;
        # no shell is involved, so tokens are never re-split or interpolated.
        cmd = ["bash", test_flavor.train_script, "--dump_folder", str(case_dir / "test_run")]
        train_args = DEFAULT_TRAIN_ARGS if test_flavor.train_args is None else test_flavor.train_args
        cmd += list(train_args)
        if test_flavor.check_loss or test_flavor.check_resume:
            cmd += list(DETERMINISTIC_ARGS)
        if override_arg:
            cmd += list(override_arg)
        # Multi-phase cases (checkpoint resume) write each phase to its own
        # TensorBoard folder; the phase suffix overrides the default "tb".
        if test_flavor.expected_steps is not None:
            cmd.append(f"--metrics.save_tb_folder=tb_phase_{idx}")
        # Human-readable rendering for the log header and error messages:
        # the runner-controlled env settings followed by the argv command.
        shown_cmd = (
            " ".join(
                f"{key}={env[key]}"
                for key in ("ASCEND_RT_VISIBLE_DEVICES", "MODULE", "CONFIG", "NGPU", "LOG_RANK")
                if key in env
                and (key in ("ASCEND_RT_VISIBLE_DEVICES", "NGPU", "LOG_RANK") or env[key] != os.environ.get(key))
            )
            + " "
            + shlex.join(cmd)
        )

        start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        # torchtitan-npu override: capture the child output and emit it as one
        # contiguous block, so concurrently running tests never interleave
        # their lines.
        result = _run_cmd(cmd, env=env, timeout=test_flavor.timeout)
        returncode = result.returncode
        captured = result.stdout or ""

        end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"===== [{test_name}] start {start_ts} end {end_ts} "
            f"flavor: {test_flavor.test_descr} (rc={returncode}) =====\n"
            f"===== [{test_name}] command: {shown_cmd} =====\n"
        )
        footer = f"===== [{test_name}] end of output (rc={returncode}) =====\n"
        _emit_block(test_name, header, captured, footer)

        if returncode != 0:
            tail = "\n".join(captured.splitlines()[-50:])
            # ``_run_cmd`` returns rc=-1 to signal a timeout.
            reason = f"timed out after {test_flavor.timeout}s" if returncode == -1 else f"rc={returncode}"
            raise RuntimeError(
                f"\nFailed test flavor: {test_flavor.test_descr} ({reason}).\n"
                f"Command: {shown_cmd}\nLast 50 lines:\n{tail}\n"
            )

        if test_flavor.verify_ema_checkpoint:
            assert_ema_checkpoint_written(case_dir / "test_run")

        _check_phase_results(test_flavor, test_name, case_dir, idx, golden_losses)

    # Resume cases compare the second phase's losses and grad norms against
    # the uninterrupted first phase (dynamic baseline, no golden file).
    if test_flavor.check_resume:
        assert test_flavor.expected_steps is not None
        compare_checkpoint_metrics(case_dir / "test_run", test_flavor.expected_steps)


def _record_case_timing(test_name: str, ngpu: int, start: float, elapsed: float) -> None:
    with _CASE_TIMINGS_LOCK:
        _CASE_TIMINGS.append(_CaseTiming(test_name, ngpu, start, elapsed))


def _run_single_test_timed(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    module: str | None = None,
    config: str | None = None,
    **kwargs,
) -> None:
    """Run one case and record its wall time for the end-of-run summary.

    The window covers everything the case costs: all of its phases plus the
    loss/checkpoint validation between them. Recording happens in ``finally``
    so a failed or timed-out case still shows up in the summary — which is
    exactly when its cost matters most.
    """
    start = time.monotonic()
    try:
        run_single_test(test_flavor, output_dir, module, config, **kwargs)
    finally:
        _record_case_timing(test_flavor.test_name, test_flavor.ngpu, start, time.monotonic() - start)


def _print_case_timings() -> None:
    """Print the per-case wall times recorded during this run.

    Emitted as the last block of the run, after the scheduling telemetry, so a
    CI reader gets each case's cost directly instead of correlating the
    per-case ``start``/``end`` headers of the output blocks by hand. Summed
    case time is the sequential equivalent; the pool window printed just above
    is the wall time actually spent.
    """
    with _CASE_TIMINGS_LOCK:
        timings = sorted(_CASE_TIMINGS, key=lambda timing: timing.start)
    if not timings:
        return
    width = max(len(timing.test_name) for timing in timings)
    log_print(f"[timing] per-case wall time ({len(timings)} case(s)):")
    for timing in timings:
        log_print(f"[timing]   {timing.test_name:<{width}}  ngpu={timing.ngpu}  {timing.elapsed:.1f}s")
    log_print(f"[timing]   sum of case times: {sum(timing.elapsed for timing in timings):.1f}s")


def _filter_tests(
    args,
    test_list: list[OverrideDefinitions],
) -> tuple[list[OverrideDefinitions], list[OverrideDefinitions]]:
    """Filter tests by --test_name / --exclude / disabled / ngpu.

    Returns (runnable, skipped_due_to_ngpu).
    """
    exclude_set = set()
    if hasattr(args, "exclude") and args.exclude:
        exclude_set = {name.strip() for name in args.exclude.split(",")}

    runnable: list[OverrideDefinitions] = []
    skipped_ngpu: list[OverrideDefinitions] = []
    for test_flavor in test_list:
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue
        if test_flavor.disabled or test_flavor.test_name in exclude_set:
            continue
        if args.ngpu < test_flavor.ngpu:
            skipped_ngpu.append(test_flavor)
            continue
        runnable.append(test_flavor)
    return runnable, skipped_ngpu


def run_tests(
    args,
    test_list: list[OverrideDefinitions],
    module=None,
    config=None,
    *,
    parallel: bool = True,
):
    """Run integration cases using Torchtitan's flow plus golden checks."""

    runnable, skipped_ngpu = _filter_tests(args, test_list)
    for test_flavor in skipped_ngpu:
        log_print(
            f"Skipping test {test_flavor.test_name} that requires "
            f"{test_flavor.ngpu} gpus, because --ngpu arg is {args.ngpu}"
        )

    failed_tests: list[tuple[str, str]] = []

    if parallel and runnable:
        # Schedule tests concurrently, packing them onto a fixed pool of
        # physical NPUs. A test can run as soon as ``test_flavor.ngpu`` NPUs
        # are free; the sum of in-flight test ngpu never exceeds the pool
        # size. The pool reflects the runner's real NPU visibility (an
        # inherited ASCEND_RT_VISIBLE_DEVICES allocation, else the CANN
        # runtime's device census) — never a fabricated id range.
        pool_ids = _resolve_npu_pool_ids(args.ngpu)
        pool = GPUPool(pool_ids)
        # A pool smaller than --ngpu (runtime-capped, or a narrower CI
        # allocation) can strand a case that passed the --ngpu filter:
        # acquire() would block forever waiting for NPUs that do not exist.
        # Skip such cases explicitly instead of deadlocking.
        for test_flavor in [t for t in runnable if t.ngpu > len(pool_ids)]:
            log_print(
                f"Skipping test {test_flavor.test_name} that requires "
                f"{test_flavor.ngpu} gpus, because only {len(pool_ids)} NPU(s) "
                "are usable by the parallel scheduler"
            )
            runnable.remove(test_flavor)
        # Submit largest-first so the very first wave packs efficiently and
        # avoids head-of-line blocking by an oversized test arriving late.
        # NOTE: this only deterministically orders the *first* batch; once
        # workers start finishing at different times, subsequent acquisition
        # order is driven by completion times, not by ``ngpu``.
        scheduled = sorted(runnable, key=lambda t: -t.ngpu)
        # torchtitan-npu override: give every case its own worker instead of
        # upstream's min(cases, ngpu). A case blocked inside acquire() holds
        # no NPUs, so a blocked big case must not consume an executor slot
        # either: on a 2-NPU allocation the upstream cap let one blocked
        # 2-NPU case starve its worker while queued 1-NPU cases serialized
        # behind it (Phase-2 full-suite canary: wall ~= sum of durations).
        # Threads blocked on the pool are cheap; the pool alone gates how
        # many training processes touch NPUs at once.
        max_workers = max(1, len(scheduled))

        def _runner(test_flavor: OverrideDefinitions) -> None:
            gpus = pool.acquire(test_flavor.ngpu)
            log_print(f"[parallel] {test_flavor.test_name}: acquired NPUs {gpus} (ngpu={test_flavor.ngpu})")
            try:
                _run_single_test_timed(
                    test_flavor,
                    args.output_dir,
                    module,
                    config,
                    golden_file=_golden_file_for(test_flavor),
                    gpu_ids=gpus,
                )
            finally:
                pool.release(gpus)
                log_print(f"[parallel] {test_flavor.test_name}: released NPUs {gpus}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future, OverrideDefinitions] = {
                executor.submit(_runner, test_flavor): test_flavor for test_flavor in scheduled
            }
            for future, test_flavor in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    log_print(str(exc))
                    failed_tests.append((test_flavor.test_name, str(exc)))
        # Scheduling telemetry: how well the pool packed the cases.
        for line in pool.stats():
            log_print(f"[parallel] {line}")
    else:
        for test_flavor in runnable:
            try:
                _run_single_test_timed(
                    test_flavor,
                    args.output_dir,
                    module,
                    config,
                    golden_file=_golden_file_for(test_flavor),
                )
            except Exception as exc:
                log_print(str(exc))
                failed_tests.append((test_flavor.test_name, str(exc)))

    # torchtitan-npu override: per-case cost, printed as the run's last block so
    # it is still emitted when a case failed above.
    _print_case_timings()

    if failed_tests:
        failure_summary = "\n".join(f"  {name}: {error}" for name, error in failed_tests)
        raise RuntimeError(f"{len(failed_tests)} integration test(s) failed:\n{failure_summary}")
    if not runnable:
        # torchtitan-npu override: a filtered-out suite is an error in CI,
        # rather than only a warning as in the upstream runner.
        raise RuntimeError(f"No tests were run for --test_name '{args.test_name}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        help="Directory to dump results generated by tests",
    )
    # torchtitan-npu override: register the repository's DeepSeek-V4 suite.
    parser.add_argument(
        "--test_suite",
        default="deepseek_v4",
        choices=sorted(_TEST_SUITES_FUNCTION),
    )
    parser.add_argument(
        "--module",
        default="torchtitan_npu.models.deepseek_v4",
        help="Model module to use for training (default: torchtitan_npu.models.deepseek_v4). "
        "This is passed as MODULE env var to run_train.sh.",
    )
    parser.add_argument(
        "--config",
        default="deepseek_v4_debugmodel",
        help="Config function to use for training (default: deepseek_v4_debugmodel). "
        "This is passed as CONFIG env var to run_train.sh.",
    )
    parser.add_argument("--test_name", default="all")
    parser.add_argument("--ngpu", type=int, default=8)
    parser.add_argument("--exclude", default=None)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run tests concurrently, packing them onto the NPU pool. "
        "At most --ngpu NPUs are in use at any time; each test is pinned to a "
        "disjoint subset via ASCEND_RT_VISIBLE_DEVICES. "
        "Use --no-parallel to force sequential execution (default: parallel).",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list, args.module, args.config, parallel=args.parallel)


if __name__ == "__main__":
    main()
