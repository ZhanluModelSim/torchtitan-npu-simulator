#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""比较两个 Git 版本的训练 loss 和 grad norm。

每个版本在独立的 Git worktree 和输出目录中运行。默认启用确定性训练，并加载
同一份 seed checkpoint；共享 checkpoint 要求两侧模型权重兼容。

验证流程：

解析 baseline/test Git ref → 创建独立 worktree → 在 baseline worktree 中生成
共享 seed checkpoint → baseline/test 加载该 checkpoint 并运行相同步数 → 从
TensorBoard 读取每个 step 的 loss 和 grad_norm → 按 bitwise 或 close 模式判定
→ 将结果写入 comparison.json。

默认在创建 seed checkpoint 和两次训练时设置 `--training.seq-len=4096`、
`--debug.seed=42`，并启用 `--debug.deterministic`。其他序列长度通过 `--seq-len` 统一指定，
步数通过 `--steps` 指定，`--no-deterministic` 可关闭确定性算法。这些由脚本控制的选项及固定的
随机种子、TensorBoard 配置不能放入 `--baseline-options` / `--test-options`。
`--no-seed-checkpoint` 可跳过共享 checkpoint。

用法示例：

1. 比较两个 Git 分支或提交

    python scripts/numerical_alignment/loss_compare.py branch1 branch2
    python scripts/numerical_alignment/loss_compare.py HEAD COMMIT_SHA

2. 指定入口脚本

默认入口脚本为：

    examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh

通过 --runner 可指定其他入口脚本：

    python scripts/numerical_alignment/loss_compare.py main HEAD \
        --runner examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh

3. 使用不同精度比较模式

比特一致模式为默认模式：

    python scripts/numerical_alignment/loss_compare.py branch1 branch2 --mode bitwise

数值近似模式：

    python scripts/numerical_alignment/loss_compare.py branch1 branch2 --mode close

4. 比较不同训练参数或 override

    python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
        --baseline-options='--override.imports baseline.factory' \
        --test-options='--override.imports candidate.factory'

    runner 通过环境变量选择 override 组时，可用 --baseline-env / --test-env
    为两侧注入不同环境（例如 debug_deepseek_v4_1p.sh 的 USE_GOLDEN）：

    python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
        --mode close \
        --no-deterministic \
        --baseline-env USE_GOLDEN=1 \
        --test-env USE_GOLDEN=0

常用参数：

- --steps：训练步数，默认 100；不支持恢复完整训练状态的续训比较。
- --runner：相对路径分别使用两个 Git ref 中的入口脚本；绝对路径两侧共用。
- --seq-len：seed 和两侧训练的序列长度，默认 4096；例如 `--seq-len 2048`。
- --ngpus：NPU 进程数，默认 1；两侧拓扑不同时用 --baseline-ngpus /
  --test-ngpus 分别指定。
- --output-dir：结果输出目录，必须为空或尚未存在；默认在当前目录的
  outputs/compare_results/ 下创建独立的结果子目录。
- --cleanup-after-test / --no-cleanup-after-test：默认清理 seed 目录和临时
  Git worktree，保留 baseline/test 训练输出和 comparison.json；使用
  --no-cleanup-after-test 保留全部产物。
  失败时也按此开关清理；中断或异常时先终止并等待训练进程组退出。
- --baseline-env / --test-env：为 baseline/test 训练注入额外的
  KEY=VALUE[,...] 环境变量（可重复）；seed checkpoint 使用 baseline 的环境。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

METRIC_TAGS = {
    "loss": "loss_metrics/global_avg_loss",
    "grad_norm": "grad_norm",
}
DEFAULT_SEQ_LEN = 4096
FIXED_TRAINING_OPTIONS = (
    "--debug.deterministic",
    "--debug.seed=42",
    "--metrics.enable_tensorboard",
    "--metrics.log_freq=1",
)
TENSORBOARD_FOLDER = "tensorboard"
DEFAULT_RUNNER = Path("examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh")
RELATIVE_ERROR_FLOOR = 1e-4


@dataclass(frozen=True)
class MetricThresholds:
    mean_ae: float
    max_re: float
    mean_re: float


DEFAULT_THRESHOLDS = MetricThresholds(
    mean_ae=0.02,
    max_re=0.005,
    mean_re=0.005,
)
BITWISE_THRESHOLDS = MetricThresholds(
    mean_ae=0.0,
    max_re=0.0,
    mean_re=0.0,
)

ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[1;36m"
ANSI_GREEN = "\033[1;32m"
ANSI_RED = "\033[1;31m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    baseline_steps: int
    test_steps: int
    mean_ae: float
    max_re: float
    mean_re: float
    passed: bool
    failed_metrics: tuple[str, ...]
    missing_steps: tuple[int, ...]
    extra_steps: tuple[int, ...]
    baseline_missing_steps: tuple[int, ...]
    baseline_extra_steps: tuple[int, ...]
    test_missing_steps: tuple[int, ...]
    test_extra_steps: tuple[int, ...]
    first_mismatch: dict[str, float | int] | None


def _colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return text
    return f"{color}{text}{ANSI_RESET}"


def log_step(message: str) -> None:
    marker = _colorize("▶", ANSI_CYAN)
    prefix = _colorize("[loss_compare]", ANSI_BOLD)
    print(f"{marker} {prefix} {message}", flush=True)


def log_info(message: str) -> None:
    marker = _colorize("•", ANSI_DIM)
    print(f"{marker} [loss_compare] {message}", flush=True)


def log_success(message: str) -> None:
    marker = _colorize("✓", ANSI_GREEN)
    print(f"{marker} [loss_compare] {message}", flush=True)


def log_failure(message: str) -> None:
    marker = _colorize("✗", ANSI_RED)
    print(f"{marker} [loss_compare] {message}", flush=True)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def find_repo_root() -> Path:
    result = _run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        capture_output=True,
    )
    return Path(result.stdout.strip()).resolve()


def resolve_ref(repo_root: Path, ref: str) -> str:
    revision = "HEAD" if ref == "." else ref
    try:
        result = _run(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"unknown Git revision: {ref}") from error
    return result.stdout.strip()


def add_worktree(repo_root: Path, path: Path, commit: str) -> None:
    _run(
        ["git", "worktree", "add", "--quiet", "--detach", str(path), commit],
        cwd=repo_root,
    )


def remove_worktree(repo_root: Path, path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=repo_root,
        check=False,
    )


def prepare_output_dir(requested: Path | None) -> Path:
    if requested is None:
        results_root = Path.cwd() / "outputs" / "compare_results"
        results_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="loss-compare-", dir=results_root)).resolve()

    output_dir = requested.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_runner(repo_root: Path, runner: Path) -> Path:
    path = runner if runner.is_absolute() else repo_root / runner
    path = path.resolve()
    if not runner.is_absolute() and not path.is_relative_to(repo_root.resolve()):
        raise ValueError("relative runner must stay inside the worktree; use an absolute path for a shared runner")
    if not path.is_file():
        raise ValueError(f"training runner not found: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"training runner is not executable: {path}")
    return path


def _fixed_training_options(*, deterministic: bool, seq_len: int) -> list[str]:
    return [
        f"--training.seq-len={seq_len}",
        *(option for option in FIXED_TRAINING_OPTIONS if deterministic or option != "--debug.deterministic"),
    ]


def build_seed_command(
    runner: Path,
    dump_folder: Path,
    *,
    deterministic: bool = True,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> list[str]:
    # Seed creation skips parallelize_fn but still builds an optimizer. Use
    # AdamW here because DistMuon requires DTensor parameters. Only model
    # weights are saved; training keeps the runner's optimizer selection.
    return [
        str(runner),
        f"--dump_folder={dump_folder}",
        "--checkpoint.create_seed_checkpoint",
        "--checkpoint.enable",
        "--checkpoint.no-load-only",
        f"--checkpoint.folder={dump_folder / 'checkpoint'}",
        "--checkpoint.last-save-model-only",
        "--optimizer.name=AdamW",
        "--parallelism.data-parallel-replicate-degree=1",
        "--parallelism.data-parallel-shard-degree=1",
        "--parallelism.tensor-parallel-degree=1",
        "--parallelism.pipeline-parallel-degree=1",
        "--parallelism.context-parallel-degree=1",
        "--parallelism.expert-parallel-degree=1",
        *_fixed_training_options(deterministic=deterministic, seq_len=seq_len),
    ]


def build_training_command(
    runner: Path,
    dump_folder: Path,
    *,
    steps: int,
    options: str,
    seed_checkpoint: Path | None,
    deterministic: bool = True,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> list[str]:
    command = [
        str(runner),
        f"--dump_folder={dump_folder}",
        *shlex.split(options),
        *_fixed_training_options(deterministic=deterministic, seq_len=seq_len),
        f"--training.steps={steps}",
        f"--metrics.save_tb_folder={TENSORBOARD_FOLDER}",
    ]
    if seed_checkpoint is not None:
        command.extend(
            [
                "--checkpoint.enable",
                "--checkpoint.load_only",
                "--checkpoint.export_dtype=bfloat16",
                f"--checkpoint.initial_load_path={seed_checkpoint}",
            ]
        )
    return command


def _process_group_running(group_id: int) -> bool:
    # killpg(..., 0) also sees orphan zombies. Linux /proc lets us wait for
    # all live descendants without waiting forever for init to reap zombies.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
            return True
    return False


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Stop the runner, torchrun and ranks before their worktree is removed."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_running(process.pid):
            break
        time.sleep(0.1)
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    while _process_group_running(process.pid):
        time.sleep(0.1)


def _handle_sigterm(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def run_with_realtime_output(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Command: {shlex.join(command)}")
    with log_file.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                output.write(line)
                output.flush()
            return_code = process.wait()
        finally:
            _terminate_process_group(process)
            process.stdout.close()
    if return_code != 0:
        log_failure(f"Command failed with exit code {return_code}")
        raise subprocess.CalledProcessError(return_code, command)


def training_env(ngpus: int, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["NGPU"] = str(ngpus)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def extract_metrics_from_tensorboard(
    dump_folder: Path,
) -> dict[str, dict[int, float]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as error:
        raise RuntimeError("tensorboard is required; install it with `python -m pip install tensorboard`") from error

    tb_root = dump_folder / TENSORBOARD_FOLDER
    event_dirs = sorted({path.parent for path in tb_root.rglob("events.out.tfevents.*")})
    candidates: list[tuple[Path, dict[str, dict[int, float]]]] = []
    for event_dir in event_dirs:
        accumulator = event_accumulator.EventAccumulator(
            str(event_dir),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        scalar_tags = set(cast("list[str]", accumulator.Tags().get("scalars", [])))
        if not set(METRIC_TAGS.values()).issubset(scalar_tags):
            continue
        metrics = {
            name: {event.step: float(event.value) for event in accumulator.Scalars(tag)}
            for name, tag in METRIC_TAGS.items()
        }
        candidates.append((event_dir, metrics))

    if len(candidates) == 1:
        return candidates[0][1]

    rank_zero = [item for item in candidates if "rank_0" in item[0].parts]
    if len(rank_zero) == 1:
        return rank_zero[0][1]

    found = [str(path) for path, _metrics in candidates]
    raise RuntimeError(
        f"expected one TensorBoard event directory with {list(METRIC_TAGS)}, found {found} under {tb_root}"
    )


def compare_metric(
    metric: str,
    baseline: dict[int, float],
    test: dict[int, float],
    *,
    mode: str,
    thresholds: MetricThresholds,
    expected_steps: int,
) -> MetricComparison:
    baseline_steps = set(baseline)
    test_steps = set(test)
    missing_steps = tuple(sorted(baseline_steps - test_steps))
    extra_steps = tuple(sorted(test_steps - baseline_steps))
    expected = set(range(1, expected_steps + 1))
    baseline_missing_steps = tuple(sorted(expected - baseline_steps))
    baseline_extra_steps = tuple(sorted(baseline_steps - expected))
    test_missing_steps = tuple(sorted(expected - test_steps))
    test_extra_steps = tuple(sorted(test_steps - expected))
    errors: list[tuple[int, float, float, float, float]] = []
    first_mismatch: dict[str, float | int] | None = None

    for step in sorted(baseline_steps & test_steps):
        baseline_value = baseline[step]
        test_value = test[step]
        finite = math.isfinite(baseline_value) and math.isfinite(test_value)
        abs_diff = abs(test_value - baseline_value) if finite else math.inf
        denominator = max(abs(baseline_value), RELATIVE_ERROR_FLOOR)
        rel_diff = abs_diff / denominator
        errors.append((step, baseline_value, test_value, abs_diff, rel_diff))

        if mode == "bitwise" and first_mismatch is None and not (finite and test_value == baseline_value):
            first_mismatch = {
                "step": step,
                "baseline": baseline_value,
                "test": test_value,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }

    abs_errors = [item[3] for item in errors]
    rel_errors = [item[4] for item in errors]
    mean_ae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    max_re = max(rel_errors, default=0.0)
    mean_re = sum(rel_errors) / len(rel_errors) if rel_errors else 0.0
    values = {
        "mean_ae": mean_ae,
        "max_re": max_re,
        "mean_re": mean_re,
    }
    effective_thresholds = thresholds if mode == "close" else BITWISE_THRESHOLDS
    limits = asdict(effective_thresholds)
    failed_metrics = tuple(name for name, value in values.items() if value > limits[name])

    if mode == "close" and failed_metrics and errors:
        if "max_re" in failed_metrics:
            mismatch = next(item for item in errors if item[4] > thresholds.max_re)
        elif "mean_ae" in failed_metrics:
            mismatch = max(errors, key=lambda item: item[3])
        else:
            mismatch = max(errors, key=lambda item: item[4])
        step, baseline_value, test_value, abs_diff, rel_diff = mismatch
        first_mismatch = {
            "step": step,
            "baseline": baseline_value,
            "test": test_value,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
        }

    return MetricComparison(
        metric=metric,
        baseline_steps=len(baseline),
        test_steps=len(test),
        mean_ae=mean_ae,
        max_re=max_re,
        mean_re=mean_re,
        passed=not (
            baseline_missing_steps or baseline_extra_steps or test_missing_steps or test_extra_steps or failed_metrics
        ),
        failed_metrics=failed_metrics,
        missing_steps=missing_steps,
        extra_steps=extra_steps,
        baseline_missing_steps=baseline_missing_steps,
        baseline_extra_steps=baseline_extra_steps,
        test_missing_steps=test_missing_steps,
        test_extra_steps=test_extra_steps,
        first_mismatch=first_mismatch,
    )


def compare_metrics(
    baseline: dict[str, dict[int, float]],
    test: dict[str, dict[int, float]],
    *,
    mode: str,
    thresholds: MetricThresholds,
    expected_steps: int,
) -> list[MetricComparison]:
    return [
        compare_metric(
            metric,
            baseline[metric],
            test[metric],
            mode=mode,
            thresholds=thresholds,
            expected_steps=expected_steps,
        )
        for metric in METRIC_TAGS
    ]


def print_comparisons(comparisons: Sequence[MetricComparison]) -> None:
    print("\n=== Training numerical comparison ===")
    for result in comparisons:
        status = "PASS" if result.passed else "FAIL"
        status_color = ANSI_GREEN if result.passed else ANSI_RED
        print(
            f"{result.metric:<10} {_colorize(status, status_color)}  steps="
            f"{result.baseline_steps}/{result.test_steps}  "
            f"mean_ae={result.mean_ae:.8e}  "
            f"max_re={result.max_re:.8e}  "
            f"mean_re={result.mean_re:.8e}"
        )
        if result.failed_metrics:
            print(f"  failed metrics: {result.failed_metrics}")
        if result.missing_steps:
            print(f"  missing test steps: {result.missing_steps}")
        if result.extra_steps:
            print(f"  extra test steps: {result.extra_steps}")
        for side in ("baseline", "test"):
            for kind in ("missing", "extra"):
                steps = getattr(result, f"{side}_{kind}_steps")
                if steps:
                    print(f"  {side} {kind} steps against expected range: {steps}")
        if result.first_mismatch is not None:
            mismatch = result.first_mismatch
            print(
                f"  first mismatch: step={mismatch['step']} baseline={mismatch['baseline']!r} test={mismatch['test']!r}"
            )


def write_report(
    output_dir: Path,
    *,
    baseline_ref: str,
    baseline_commit: str,
    test_ref: str,
    test_commit: str,
    baseline_runner: Path,
    test_runner: Path,
    runner_scope: str,
    expected_steps: int,
    mode: str,
    thresholds: MetricThresholds,
    comparisons: Sequence[MetricComparison],
) -> None:
    report: dict[str, Any] = {
        "baseline": {"ref": baseline_ref, "commit": baseline_commit, "runner": str(baseline_runner)},
        "test": {"ref": test_ref, "commit": test_commit, "runner": str(test_runner)},
        "runner_scope": runner_scope,
        "expected_step_range": {"first": 1, "last": expected_steps},
        "mode": mode,
        "thresholds": asdict(thresholds),
        "passed": all(result.passed for result in comparisons),
        "metrics": [asdict(result) for result in comparisons],
    }
    with (output_dir / "comparison.json").open("w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def _parse_env(spec: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE[,KEY=VALUE...]`` env spec into a dict."""
    env: dict[str, str] = {}
    for item in spec.split(","):
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise argparse.ArgumentTypeError(f"invalid env spec {item!r}; expected KEY=VALUE")
        env[key] = value
    return env


def _merged_env(specs: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Merge repeatable env specs, later entries overriding earlier ones."""
    merged: dict[str, str] = {}
    for spec in specs:
        merged.update(spec)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("baseline_ref", help="baseline Git branch, tag, or commit")
    parser.add_argument("test_ref", help="test Git branch, tag, or commit")
    parser.add_argument(
        "--mode",
        choices=("bitwise", "close"),
        default="bitwise",
        help="comparison mode (default: bitwise)",
    )
    for name, default in asdict(DEFAULT_THRESHOLDS).items():
        parser.add_argument(
            f"--{name.replace('_', '-')}-threshold",
            type=float,
            default=default,
            help=f"{name} tolerance for close mode (default: {default})",
        )
    parser.add_argument(
        "--steps", type=int, default=100, help="require metrics for every step from 1 to N (default: 100)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="sequence length for seed and both runs (default: 4096)"
    )
    parser.add_argument("--ngpus", type=int, default=1, help="NPU processes for both runs (default)")
    parser.add_argument(
        "--baseline-ngpus",
        type=int,
        default=None,
        help="NPU processes for the baseline run (overrides --ngpus; use when "
        "the two parallelism topologies differ, e.g. 1D FSDP vs 2D FSDP+TP)",
    )
    parser.add_argument(
        "--test-ngpus",
        type=int,
        default=None,
        help="NPU processes for the test run (overrides --ngpus)",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help="relative path: run each ref's own script; absolute path: use one shared script",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: a unique subdirectory under ./outputs/compare_results/)",
    )
    parser.add_argument(
        "--cleanup-after-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove the seed directory and temporary Git worktrees on exit, including failures; "
        "keep baseline/test training outputs and the comparison report "
        "(default: enabled; --no-cleanup-after-test keeps all artifacts)",
    )
    parser.add_argument("--baseline-options", default="")
    parser.add_argument("--test-options", default="")
    parser.add_argument(
        "--baseline-env",
        type=_parse_env,
        action="append",
        default=[],
        help="extra KEY=VALUE[,...] environment for the baseline run (repeatable)",
    )
    parser.add_argument(
        "--test-env",
        type=_parse_env,
        action="append",
        default=[],
        help="extra KEY=VALUE[,...] environment for the test run (repeatable)",
    )
    parser.add_argument(
        "--no-seed-checkpoint",
        action="store_true",
        help="initialize both runs from deterministic seeds instead of one checkpoint",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="do not enable deterministic algorithms (required by unsupported fused ops)",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    managed_keys = {
        option.partition("=")[0].replace("_", "-")
        for option in _fixed_training_options(deterministic=True, seq_len=args.seq_len)
    }
    managed_keys.update(("--training.steps", "--metrics.save-tb-folder"))
    option_hints = {
        "--training.seq-len": "use --seq-len for seed and both runs",
        "--training.steps": "use --steps for both runs",
        "--debug.deterministic": "use --no-deterministic to disable deterministic algorithms for both runs",
        "--debug.seed": "the seed is fixed at 42 for seed creation and both runs",
    }
    for name in ("baseline", "test"):
        options = getattr(args, f"{name}_options")
        for option in shlex.split(options):
            key = option.partition("=")[0].replace("_", "-").replace(".no-", ".")
            if key in managed_keys:
                hint = option_hints.get(
                    key, "TensorBoard settings are fixed to record every step in the comparison output"
                )
                parser.error(f"--{name}-options must not set {option.partition('=')[0]}; {hint}")
    if args.ngpus <= 0:
        parser.error("--ngpus must be positive")
    for name in ("baseline", "test"):
        ngpus = getattr(args, f"{name}_ngpus")
        if ngpus is not None and ngpus <= 0:
            parser.error(f"--{name}-ngpus must be positive")
    if any(getattr(args, f"{name}_threshold") < 0 for name in asdict(DEFAULT_THRESHOLDS)):
        parser.error("comparison thresholds must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    output_dir = prepare_output_dir(args.output_dir)

    log_step("Resolving Git revisions")
    baseline_commit = resolve_ref(repo_root, args.baseline_ref)
    test_commit = resolve_ref(repo_root, args.test_ref)
    thresholds = MetricThresholds(
        mean_ae=args.mean_ae_threshold,
        max_re=args.max_re_threshold,
        mean_re=args.mean_re_threshold,
    )

    log_info(f"Baseline: {args.baseline_ref} ({baseline_commit})")
    log_info(f"Test: {args.test_ref} ({test_commit})")
    base_n = args.baseline_ngpus or args.ngpus
    test_n = args.test_ngpus or args.ngpus
    if base_n == test_n:
        log_info(f"Mode: {args.mode}; steps: {args.steps}; NPU processes: {args.ngpus}")
    else:
        log_info(f"Mode: {args.mode}; steps: {args.steps}; NPU processes: baseline={base_n}, test={test_n}")
    log_info(f"Sequence length: {args.seq_len}")
    log_info(f"Deterministic algorithms: {not args.no_deterministic}")
    log_info(f"Output: {output_dir}")

    worktree_root = Path(tempfile.mkdtemp(prefix="loss-compare-worktrees-"))
    baseline_worktree = worktree_root / "baseline"
    test_worktree = worktree_root / "test"
    added_worktrees: list[Path] = []
    original_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        log_step("Creating isolated Git worktrees")
        add_worktree(repo_root, baseline_worktree, baseline_commit)
        added_worktrees.append(baseline_worktree)
        add_worktree(repo_root, test_worktree, test_commit)
        added_worktrees.append(test_worktree)
        log_success("Git worktrees are ready")
        baseline_runner = resolve_runner(baseline_worktree, args.runner)
        test_runner = resolve_runner(test_worktree, args.runner)
        runner_scope = "shared" if args.runner.is_absolute() else "per-ref"
        log_info(f"Runner scope: {runner_scope}")
        log_info(f"Baseline runner: {baseline_runner}")
        log_info(f"Test runner: {test_runner}")
        log_info(f"Expected metric steps: 1..{args.steps}")

        seed_checkpoint = None
        if not args.no_seed_checkpoint:
            log_step("Creating shared seed checkpoint")
            seed_dump = output_dir / "seed"
            run_with_realtime_output(
                build_seed_command(
                    baseline_runner,
                    seed_dump,
                    deterministic=not args.no_deterministic,
                    seq_len=args.seq_len,
                ),
                cwd=baseline_worktree,
                env=training_env(1, _merged_env(args.baseline_env)),
                log_file=seed_dump / "training.log",
            )
            seed_checkpoint = seed_dump / "checkpoint" / "step-0"
            if not seed_checkpoint.is_dir():
                raise RuntimeError(f"seed checkpoint was not created: {seed_checkpoint}")
            log_success(f"Seed checkpoint is ready: {seed_checkpoint}")
        else:
            log_info("Shared seed checkpoint disabled; using deterministic seeds")

        log_step(f"Running baseline training (steps={args.steps})")
        baseline_dump = output_dir / "baseline"
        run_with_realtime_output(
            build_training_command(
                baseline_runner,
                baseline_dump,
                steps=args.steps,
                options=args.baseline_options,
                seed_checkpoint=seed_checkpoint,
                deterministic=not args.no_deterministic,
                seq_len=args.seq_len,
            ),
            cwd=baseline_worktree,
            env=training_env(
                args.baseline_ngpus if args.baseline_ngpus is not None else args.ngpus,
                _merged_env(args.baseline_env),
            ),
            log_file=baseline_dump / "training.log",
        )
        log_success("Baseline training completed")

        log_step("Reading baseline TensorBoard metrics")
        baseline_metrics = extract_metrics_from_tensorboard(baseline_dump)
        log_success(
            "Baseline metrics loaded: "
            + ", ".join(f"{name}={len(values)} steps" for name, values in baseline_metrics.items())
        )

        log_step(f"Running test training (steps={args.steps})")
        test_dump = output_dir / "test"
        run_with_realtime_output(
            build_training_command(
                test_runner,
                test_dump,
                steps=args.steps,
                options=args.test_options,
                seed_checkpoint=seed_checkpoint,
                deterministic=not args.no_deterministic,
                seq_len=args.seq_len,
            ),
            cwd=test_worktree,
            env=training_env(
                args.test_ngpus if args.test_ngpus is not None else args.ngpus,
                _merged_env(args.test_env),
            ),
            log_file=test_dump / "training.log",
        )
        log_success("Test training completed")

        log_step("Reading test TensorBoard metrics")
        test_metrics = extract_metrics_from_tensorboard(test_dump)
        log_success(
            "Test metrics loaded: " + ", ".join(f"{name}={len(values)} steps" for name, values in test_metrics.items())
        )

        log_step("Comparing loss and grad_norm")
        comparisons = compare_metrics(
            baseline_metrics,
            test_metrics,
            mode=args.mode,
            thresholds=thresholds,
            expected_steps=args.steps,
        )
        print_comparisons(comparisons)

        log_step("Writing comparison report")
        write_report(
            output_dir,
            baseline_ref=args.baseline_ref,
            baseline_commit=baseline_commit,
            test_ref=args.test_ref,
            test_commit=test_commit,
            baseline_runner=baseline_runner,
            test_runner=test_runner,
            runner_scope=runner_scope,
            expected_steps=args.steps,
            mode=args.mode,
            thresholds=(thresholds if args.mode == "close" else BITWISE_THRESHOLDS),
            comparisons=comparisons,
        )
        log_success(f"Report written: {output_dir / 'comparison.json'}")
        if not all(result.passed for result in comparisons):
            log_failure("Comparison failed")
            raise SystemExit(1)
        log_success("Comparison passed")
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        if args.cleanup_after_test:
            log_step("Cleaning seed directory and temporary Git worktrees")
            for worktree in reversed(added_worktrees):
                remove_worktree(repo_root, worktree)
            shutil.rmtree(worktree_root, ignore_errors=True)
            seed_dir = output_dir / "seed"
            if seed_dir.exists():
                shutil.rmtree(seed_dir)
            log_success("Seed directory and temporary Git worktrees removed")
        else:
            log_info(f"Cleanup disabled; training artifacts kept in: {output_dir}")
            log_info(f"Git worktrees kept in: {worktree_root}")
        log_info("Consider using the .agents/skills/training-log-visualization skill to plot training log curves.")


if __name__ == "__main__":
    main()
