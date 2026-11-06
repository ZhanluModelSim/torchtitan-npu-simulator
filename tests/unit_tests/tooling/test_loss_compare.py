# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the comparison CLI with local Git refs and recorded CPU metrics."""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.numerical_alignment import loss_compare

pytestmark = pytest.mark.tooling
CLI = Path(loss_compare.__file__).resolve()


def test_loss_compare_supports_independent_sequence_apis_and_upstreams(tmp_path):
    (tmp_path / "torchtitan").mkdir()
    (tmp_path / "torchtitan" / "__init__.py").touch()
    common = dict(steps=100, options="", seed_checkpoint=None, seq_len=512)
    old = loss_compare.build_training_command(tmp_path / "run.sh", tmp_path / "old", training_api="legacy", **common)
    new = loss_compare.build_training_command(tmp_path / "run.sh", tmp_path / "new", **common)

    assert "--training.seq-len=512" in old
    assert "--training.max-context-length=512" in new
    assert not any(arg.startswith("--training.seq-len") for arg in new)
    env = loss_compare.training_env(2, {"PYTHONPATH": "/independent/npu"}, upstream=tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path.resolve()), "/independent/npu"]


def test_loss_compare_bitwise_rejects_signed_zero():
    comparisons = loss_compare.compare_metrics(
        {"loss": {1: 0.0}, "grad_norm": {1: 1.0}},
        {"loss": {1: -0.0}, "grad_norm": {1: 1.0}},
        mode="bitwise", thresholds=loss_compare.DEFAULT_THRESHOLDS, expected_steps=1,
    )
    assert not comparisons[0].passed


def test_loss_compare_native_entrypoint_rejects_custom_train_file(monkeypatch):
    monkeypatch.setenv("TRAIN_FILE", "custom.training")

    with pytest.raises(ValueError, match="unset the custom TRAIN_FILE"):
        loss_compare.training_env(8, config_api="native")


@pytest.fixture
def runner_repo(tmp_path):
    """Two refs differ only in their runner's emitted scalar value, not a model."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Tooling test")
    git("config", "user.email", "tooling@example.com")
    git("config", "core.hooksPath", "/dev/null")
    (repo / "emit_metrics.py").write_text(
        """import argparse
import os
from pathlib import Path
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
p = argparse.ArgumentParser()
p.add_argument('--dump_folder', type=Path)
p.add_argument('--training.steps', type=int, dest='steps')
p.add_argument('--value', type=float)
p.add_argument('--recorded-steps')
p.add_argument('--checkpoint.folder', '--comparison-checkpoint-folder', type=Path, dest='checkpoint_folder')
p.add_argument('--checkpoint.create_seed_checkpoint', '--comparison-create-seed-checkpoint',
               action='store_true', dest='create_seed_checkpoint')
args, _ = p.parse_known_args()
if args.create_seed_checkpoint:
    (args.checkpoint_folder / 'step-0').mkdir(parents=True)
    print('seed-ngpu=' + os.environ.get('NGPU', 'unset'))
    raise SystemExit(0)
print('runner-ngpu=' + os.environ.get('NGPU', 'unset'))
print('runner-cwd=' + str(Path.cwd()))
steps = range(1, args.steps + 1) if args.recorded_steps is None else map(int, args.recorded_steps.split(','))
w = EventFileWriter(str(args.dump_folder / 'tensorboard' / 'rank_0'))
try:
    for step in steps:
        for tag in ('loss_metrics/global_avg_loss', 'grad_norm'):
            w.add_event(Event(step=step, summary=Summary(value=[Summary.Value(tag=tag, simple_value=args.value)])))
finally:
    w.close()
print('runner-value=' + str(args.value), flush=True)
"""
    )
    runner = repo / "runner.sh"
    refs = []
    for value in (1, 2):
        runner.write_text(
            f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$(dirname "$0")/emit_metrics.py" --value={value} "$@"\n'
        )
        runner.chmod(0o755)
        git("add", ".")
        git("commit", "-qm", f"Runner scalar {value}")
        refs.append(git("rev-parse", "HEAD"))
    return repo, refs


def _compare(repo, refs, output, runner, *options):
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            *refs,
            "--no-seed-checkpoint",
            "--steps=4",
            "--runner",
            str(runner),
            "--output-dir",
            str(output),
            *options,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=40,
    )


def _write_metric_runner(repo, name, value, *, ngpus=None):
    runner = repo / name
    environment = f'export NGPU="${{NGPU:-{ngpus}}}"\n' if ngpus is not None else ""
    runner.write_text(
        f'#!/bin/bash\n{environment}echo runner-name={name}\n'
        f'exec {shlex.quote(sys.executable)} emit_metrics.py --value={value} "$@"\n'
    )
    runner.chmod(0o755)
    return runner


def test_loss_compare_selects_independent_config_apis_and_source_roots(tmp_path, runner_repo):
    # This checks the tooling protocol using recorded metrics, not NPU training
    # or checkpoint compatibility (which require the real example runners).
    repo, _ = runner_repo
    emitter = repo / "emit_metrics.py"
    source = emitter.read_text()
    refs = []
    upstreams = []
    for api in ("legacy", "native"):
        upstream = tmp_path / api
        upstream.mkdir()
        (upstream / "torchtitan").mkdir()
        (upstream / "torchtitan" / "__init__.py").touch()
        (upstream / "selected_upstream.py").write_text(f"API = {api!r}\n")
        upstreams.append(upstream)
        protocol = f'''import os, sys
from pathlib import Path
from selected_upstream import API
assert API == {api!r}
assert Path(sys.path[0]).resolve() == Path.cwd()
keys = [arg.split('=')[0] for arg in sys.argv[1:]]
if API == 'native':
    assert os.environ['TRAIN_FILE'] == 'scripts.numerical_alignment.train'
    assert not any(key.startswith(('--checkpoint.', '--optimizer.')) for key in keys)
    assert '--comparison-initial-load-path' in keys
else:
    assert not any(key.startswith(('--comparison-', '--optim.')) for key in keys)
    assert '--checkpoint.create_seed_checkpoint' in keys or '--checkpoint.initial_load_path' in keys
print('selected-api=' + API)
'''
        emitter.write_text(protocol + source)
        subprocess.run(["git", "add", "emit_metrics.py"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", f"Recorded {api} control protocol"], cwd=repo, check=True)
        refs.append(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    output = tmp_path / "comparison"

    result = subprocess.run(
        [sys.executable, str(CLI), *refs, "--runner=runner.sh", "--steps=2",
         "--baseline-config-api=legacy", "--test-config-api=native",
         f"--baseline-upstream={upstreams[0]}", f"--test-upstream={upstreams[1]}", f"--output-dir={output}"],
        cwd=repo, text=True, capture_output=True, timeout=40,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "comparison.json").read_text())
    assert report["passed"]
    for side, api in (("baseline", "legacy"), ("test", "native")):
        assert f"selected-api={api}" in (output / side / "training.log").read_text()
        assert report["runtime_options"][side]["config_api"] == api


@pytest.mark.parametrize(
    ("inherited_ngpu", "options", "baseline_ngpu", "test_ngpu", "baseline_config", "test_config"),
    [
        (None, (), 8, 4, (None, None, "runner"), (None, None, "runner")),
        ("6", (), 6, 6, (None, "6", "environment"), (None, "6", "environment")),
        ("6", ("--ngpus=2",), 2, 2, (2, "2", "override"), (2, "2", "override")),
        (
            "6",
            ("--ngpus=2", "--baseline-ngpus=3", "--test-ngpus=5"),
            3,
            5,
            (3, "3", "override"),
            (5, "5", "override"),
        ),
        (None, ("--baseline-ngpus=3",), 3, 4, (3, "3", "override"), (None, None, "runner")),
        (
            "6",
            ("--ngpus=2", "--baseline-env=NGPU=9", "--test-env=NGPU=7"),
            2,
            2,
            (2, "2", "override"),
            (2, "2", "override"),
        ),
        (
            "6",
            ("--baseline-env=NGPU=9", "--test-env=NGPU=7"),
            9,
            7,
            (None, "9", "environment"),
            (None, "7", "environment"),
        ),
    ],
    ids=[
        "runner_defaults",
        "inherited",
        "shared_override",
        "per_side_overrides",
        "one_side",
        "env_conflict",
        "per_side_env",
    ],
)
def test_loss_compare_preserves_runner_card_counts_unless_overridden(
    tmp_path,
    runner_repo,
    monkeypatch,
    inherited_ngpu,
    options,
    baseline_ngpu,
    test_ngpu,
    baseline_config,
    test_config,
):
    repo, _refs = runner_repo
    _write_metric_runner(repo, "base.sh", 1, ngpus=8)
    _write_metric_runner(repo, "test.sh", 1, ngpus=4)
    monkeypatch.delenv("NGPU", raising=False)
    if inherited_ngpu is not None:
        monkeypatch.setenv("NGPU", inherited_ngpu)
    output = tmp_path / "comparison"

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--baseline-runner=base.sh",
            "--test-runner=test.sh",
            "--steps=2",
            "--output-dir",
            str(output),
            *options,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=40,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "seed-ngpu=1\n" in result.stdout
    report = json.loads((output / "comparison.json").read_text())
    for side, ngpu, config in (("baseline", baseline_ngpu, baseline_config), ("test", test_ngpu, test_config)):
        assert f"runner-ngpu={ngpu}\n" in (output / side / "training.log").read_text()
        override, ngpu_env, source = config
        assert report[side]["npu_processes"] == {"override": override, "ngpu_env": ngpu_env, "source": source}


@pytest.mark.parametrize("test_value", [1, 2], ids=["matching", "different"])
def test_loss_compare_runs_uncommitted_scripts_with_shared_seed(tmp_path, runner_repo, test_value):
    repo, refs = runner_repo
    base_runner = _write_metric_runner(repo, "base.sh", 1)
    test_runner = _write_metric_runner(repo, "test.sh", test_value)
    # The emitter must also come from the working tree, rather than HEAD.
    emitter = repo / "emit_metrics.py"
    emitter.write_text(emitter.read_text() + "\nprint('working-tree-emitter')\n")
    status_before = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    output = tmp_path / "comparison"

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--baseline-runner=base.sh",
            "--test-runner=test.sh",
            "--steps=2",
            "--output-dir",
            str(output),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=40,
    )

    assert result.returncode == (0 if test_value == 1 else 1), result.stdout + result.stderr
    assert "seed-ngpu=1\n" in result.stdout
    report = json.loads((output / "comparison.json").read_text())
    assert report["runner_scope"] == "working-tree"
    assert report["passed"] is (test_value == 1)
    assert "Current-workspace runner=" not in result.stdout
    for side, runner, value in (("baseline", base_runner, 1), ("test", test_runner, test_value)):
        assert report[side]["ref"] == "working-tree"
        assert report[side]["commit"] == refs[-1]
        assert report[side]["runner"] == str(runner)
        log = (output / side / "training.log").read_text()
        assert f"runner-name={runner.name}" in log
        assert f"runner-value={value}.0" in log
        assert "working-tree-emitter" in log
        assert runner.is_file()
    assert not (output / "seed").exists()
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True) == status_before
    trees = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=repo, text=True)
    assert trees.count("worktree ") == 1


def test_loss_compare_runs_working_tree_scripts_against_selected_refs(tmp_path, runner_repo):
    repo, _refs = runner_repo
    emitter = repo / "emit_metrics.py"
    source = emitter.read_text()
    refs = []
    for version in ("baseline", "test"):
        emitter.write_text(source + f"\nprint('code-version={version}')\n")
        subprocess.run(["git", "add", "emit_metrics.py"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", f"Emitter {version}"], cwd=repo, check=True)
        refs.append(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    emitter.write_text(source + "\nprint('code-version=current')\n")
    base_runner = _write_metric_runner(repo, "base.sh", 1)
    test_runner = _write_metric_runner(repo, "test.sh", 1)
    status_before = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    output = tmp_path / "comparison"

    result = _compare(
        repo,
        refs,
        output,
        Path("missing-default.sh"),
        "--baseline-runner=base.sh",
        "--test-runner=test.sh",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "comparison.json").read_text())
    assert report["passed"]
    for side, runner, commit in (("baseline", base_runner, refs[0]), ("test", test_runner, refs[1])):
        assert report[side]["ref"] == commit
        assert report[side]["commit"] == commit
        assert report[side]["runner"] == str(runner)
        log = (output / side / "training.log").read_text()
        cwd = next(line.removeprefix("runner-cwd=") for line in log.splitlines() if line.startswith("runner-cwd="))
        warning = next(line for line in result.stdout.splitlines() if f"[{side}] Current-workspace runner=" in line)
        assert "WARNING" in warning
        assert f"runner={runner}; target commit={commit}; expected training cwd={cwd}." in warning
        assert Path(cwd) != repo
        assert "actual source selection is not verified" in warning
        assert f"runner-name={runner.name}" in log
        assert f"code-version={side}" in log
        assert "code-version=current" not in log
        assert runner.is_file()
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True) == status_before


@pytest.mark.parametrize("side", ["baseline", "test"])
def test_loss_compare_keeps_shared_runner_as_other_side_default(tmp_path, runner_repo, side):
    repo, refs = runner_repo
    runner = _write_metric_runner(repo, "custom.sh", 1)
    output = tmp_path / "comparison"

    result = _compare(repo, [refs[0], refs[0]], output, Path("runner.sh"), f"--{side}-runner={runner.name}")

    assert result.returncode == 0, result.stdout + result.stderr
    custom_side, default_side = ("baseline", "test") if side == "baseline" else ("test", "baseline")
    assert f"[{custom_side}] Current-workspace runner=" in result.stdout
    assert f"[{default_side}] Current-workspace runner=" not in result.stdout
    assert "runner-name=custom.sh" in (output / custom_side / "training.log").read_text()
    assert "runner-value=1.0" in (output / default_side / "training.log").read_text()


@pytest.mark.parametrize("side", ["baseline", "test"])
@pytest.mark.parametrize(
    ("options", "hint"),
    [
        ("--training.seq_len 2048", "use --seq-len"),
        ("--training.seq-len=2048", "use --seq-len"),
        ("--training.max-context-length=2048", "use --seq-len"),
        ("--training.steps=5", "use --steps"),
        ("--debug.seed 123", "fixed at 42"),
        ("--debug.deterministic", "use --no-deterministic"),
        ("--debug.no_deterministic", "use --no-deterministic"),
        ("--metrics.enable_tensorboard", "TensorBoard settings are fixed"),
        ("--metrics.no-enable-tensorboard", "TensorBoard settings are fixed"),
        ("--metrics.log_freq=10", "TensorBoard settings are fixed"),
        ("--metrics.save-tb-folder=elsewhere", "TensorBoard settings are fixed"),
    ],
    ids=[
        "seq_len_space",
        "seq_len_equals",
        "max_context_length",
        "steps",
        "seed",
        "deterministic",
        "no_deterministic",
        "enable_tensorboard",
        "disable_tensorboard",
        "log_freq",
        "tb_folder",
    ],
)
def test_loss_compare_rejects_managed_options_before_creating_output(tmp_path, side, options, hint):
    output = tmp_path / "comparison"

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "HEAD",
            "HEAD",
            f"--{side}-options={options}",
            "--no-deterministic",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert f"--{side}-options must not set" in result.stderr
    assert hint in result.stderr
    assert not output.exists()


def test_loss_compare_forwards_unmanaged_options(tmp_path, runner_repo):
    repo, refs = runner_repo
    output = tmp_path / "comparison"

    result = _compare(
        repo,
        [refs[0], refs[0]],
        output,
        Path("runner.sh"),
        "--seq-len=2048",
        "--no-deterministic",
        "--baseline-options=--value=3",
        "--test-options=--value=3",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "comparison.json").read_text())
    assert report["passed"]
    for side in ("baseline", "test"):
        assert "runner-value=3.0" in (output / side / "training.log").read_text()


@pytest.mark.parametrize("shared", [False, True], ids=["per_ref_runner", "shared_absolute_runner"])
def test_loss_compare_uses_requested_runner_scope(tmp_path, runner_repo, shared):
    repo, refs = runner_repo
    runner = repo / "runner.sh" if shared else Path("runner.sh")
    output = tmp_path / "comparison"

    result = _compare(repo, refs, output, runner)

    report = json.loads((output / "comparison.json").read_text())
    assert result.returncode == (0 if shared else 1), result.stdout + result.stderr
    assert report["runner_scope"] == ("shared" if shared else "per-ref")
    for side in ("baseline", "test"):
        assert (f"[{side}] Current-workspace runner=" in result.stdout) is shared
    assert report["passed"] is shared
    assert "runner-value=2.0" in (output / "test/training.log").read_text()
    assert f"runner-value={2 if shared else 1}.0" in (output / "baseline/training.log").read_text()
    if shared:
        assert report["baseline"]["runner"] == report["test"]["runner"] == str(runner)
    else:
        assert report["metrics"][0]["first_mismatch"] == {
            "step": 1,
            "baseline": 1.0,
            "test": 2.0,
            "abs_diff": 1.0,
            "rel_diff": 1.0,
        }


@pytest.mark.parametrize(
    ("recorded", "mode", "passed", "missing", "extra"),
    [
        ("1,2,3,4", "bitwise", True, [], []),
        ("1,2,3", "bitwise", False, [4], []),
        ("1,2,4", "close", False, [3], []),
        ("3,4", "close", False, [1, 2], []),
        ("0,1,2,3,4", "bitwise", False, [], [0]),
    ],
    ids=["complete", "truncated", "gap", "resumed", "unexpected_step"],
)
def test_loss_compare_requires_complete_metric_range(tmp_path, runner_repo, recorded, mode, passed, missing, extra):
    repo, refs = runner_repo
    output = tmp_path / "comparison"
    options = f"--recorded-steps={recorded}"

    result = _compare(
        repo,
        [refs[0], refs[0]],
        output,
        Path("runner.sh"),
        f"--mode={mode}",
        f"--baseline-options={options}",
        f"--test-options={options}",
    )

    report = json.loads((output / "comparison.json").read_text())
    assert result.returncode == (0 if passed else 1), result.stdout + result.stderr
    assert report["passed"] is passed
    assert report["expected_step_range"] == {"first": 1, "last": 4}
    for metric in report["metrics"]:
        assert metric["passed"] is passed
        for side in ("baseline", "test"):
            assert metric[f"{side}_missing_steps"] == missing
            assert metric[f"{side}_extra_steps"] == extra


def test_loss_compare_empty_metrics_cannot_pass():
    result = loss_compare.compare_metric(
        "loss",
        {},
        {},
        mode="bitwise",
        thresholds=loss_compare.DEFAULT_THRESHOLDS,
        expected_steps=100,
    )

    assert not result.passed
    assert result.baseline_missing_steps == result.test_missing_steps == tuple(range(1, 101))


def _wait_for_file(path):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text():
            return
        time.sleep(0.05)
    raise AssertionError(f"runner did not create {path}")


@pytest.fixture
def sleeping_runner(tmp_path):
    """A real process tree whose descendants ignore TERM to exercise escalation."""
    pid_file = tmp_path / "pids.json"
    child_source = (
        "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
        f"Path({str(pid_file)!r}).write_text(json.dumps([os.getpgrp(),os.getpid(),child.pid])); time.sleep(60)"
    )
    runner = tmp_path / "sleeping_runner.py"
    runner.write_text(
        f"#!{sys.executable}\nimport subprocess,sys,time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "print('READY', flush=True)\ntime.sleep(60)\n"
    )
    runner.chmod(0o755)
    try:
        yield runner, pid_file
    finally:
        if pid_file.is_file():
            try:
                os.killpg(json.loads(pid_file.read_text())[0], signal.SIGKILL)
            except ProcessLookupError:
                pass


def _assert_processes_stopped(pids):
    # An orphan already in Z state has exited and cannot use its removed worktree.
    for pid in pids:
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().split(")", 1)[1].split()[0] == "Z", f"process {pid} still running"


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["interrupt", "terminate"])
def test_loss_compare_signal_stops_descendants_before_cleanup(tmp_path, runner_repo, sleeping_runner, signum):
    repo, refs = runner_repo
    runner, pid_file = sleeping_runner
    output = tmp_path / "comparison"
    with (tmp_path / "console.log").open("w") as console:
        process = subprocess.Popen(
            [
                sys.executable,
                str(CLI),
                refs[0],
                refs[0],
                "--runner",
                str(runner),
                "--no-seed-checkpoint",
                "--output-dir",
                str(output),
            ],
            cwd=repo,
            stdout=console,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_file(pid_file)
            process.send_signal(signum)
            code = process.wait(timeout=20)

            assert code != 0
            _assert_processes_stopped(json.loads(pid_file.read_text()))
            trees = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=repo, text=True)
            assert trees.count("worktree ") == 1
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def test_loss_compare_output_exception_stops_descendants(tmp_path, sleeping_runner, monkeypatch):
    runner, pid_file = sleeping_runner

    def print_with_error(message, **_kwargs):
        if message.startswith("READY"):
            _wait_for_file(pid_file)
            raise OSError("output unavailable")

    monkeypatch.setattr(loss_compare, "print", print_with_error, raising=False)

    with pytest.raises(OSError, match="output unavailable"):
        loss_compare.run_with_realtime_output(
            [str(runner)],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_file=tmp_path / "training.log",
        )

    _assert_processes_stopped(json.loads(pid_file.read_text()))
