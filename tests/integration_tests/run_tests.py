#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/cc286a63599e42480a07928cc362e514ae448a85/tests/integration_tests/run_tests.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

# torchtitan-npu override: the runner consumes the case definition at runtime.
from tests.integration_tests import OverrideDefinitions  # noqa: TC001
from tests.integration_tests.deepseek_v4 import build_deepseek_v4_test_list
from tests.integration_tests.loss_compare import (
    assert_losses_equal,
    extract_losses_from_tensorboard,
    log_print,
    read_losses_from_file,
)

# torchtitan-npu override: this runner currently registers
# the DeepSeek-V4 NPU suite instead of Torchtitan's generic suites.
_TEST_SUITES_FUNCTION = {
    "deepseek_v4": build_deepseek_v4_test_list,
}
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
)
DETERMINISTIC_ARGS = (
    "--debug.deterministic",
    "--debug.seed=42",
)


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    module: str | None = None,
    config: str | None = None,
    *,
    golden_file: str | Path | None = None,
) -> None:
    """Run one case, optionally validating its exact loss trajectory."""

    test_name = test_flavor.test_name
    case_dir = Path(output_dir) / test_name
    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))
    golden_losses = None
    if test_flavor.check_loss:
        if golden_file is None:
            raise ValueError(f"golden_file is required when check_loss=True for {test_name}")
        golden_losses = read_losses_from_file(golden_file)

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = ""
        if module is not None:
            cmd += f"MODULE={module} "
        if config is not None:
            cmd += f"CONFIG={config} "
        if test_flavor.env_vars:
            cmd += " ".join(f"{key}={value}" for key, value in test_flavor.env_vars.items()) + " "
        cmd += f"NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} bash scripts/run_train.sh"
        cmd += f" --dump_folder {case_dir / 'test_run'}"
        cmd += " " + " ".join(DEFAULT_TRAIN_ARGS)
        if test_flavor.check_loss:
            cmd += " " + " ".join(DETERMINISTIC_ARGS)
        if override_arg:
            cmd += " " + " ".join(override_arg)

        # torchtitan-npu override: keep training output live and validate each
        # variation against the checked-in reference.
        subprocess.run(cmd, shell=True, check=True)

        if not test_flavor.check_loss:
            continue

        test_losses = extract_losses_from_tensorboard(case_dir / "test_run", "tb")
        assert golden_losses is not None
        # torchtitan-npu override: on mismatch, print losses in the
        # reference-file format to simplify deliberate regeneration.
        try:
            assert_losses_equal(golden_losses, test_losses)
        except AssertionError:
            print(f"[GOLDEN_MISMATCH] {test_name} — dumping actual losses for regeneration:")
            for step in sorted(test_losses):
                print(f"[GOLDEN_MISMATCH] {step} {test_losses[step]}")
            raise


def run_tests(args, test_list: list[OverrideDefinitions], module=None, config=None):
    """Run integration cases using Torchtitan's flow plus golden checks."""

    exclude_set = set()
    if args.exclude:
        exclude_set = {name.strip() for name in args.exclude.split(",")}

    ran_any_test = False
    failed_tests: list[tuple[str, str]] = []
    for test_flavor in test_list:
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue
        if test_flavor.disabled or test_flavor.test_name in exclude_set:
            continue
        if args.ngpu < test_flavor.ngpu:
            log_print(
                f"Skipping test {test_flavor.test_name} that requires "
                f"{test_flavor.ngpu} gpus, because --ngpu arg is {args.ngpu}"
            )
            continue
        ran_any_test = True
        try:
            golden_file = _GOLDEN_DIR / f"{test_flavor.test_name}.txt" if test_flavor.check_loss else None
            run_single_test(
                test_flavor,
                args.output_dir,
                module,
                config,
                golden_file=golden_file,
            )
        except Exception as exc:
            failed_tests.append((test_flavor.test_name, str(exc)))

    if failed_tests:
        failure_summary = "\n".join(f"  {name}: {error}" for name, error in failed_tests)
        raise RuntimeError(f"{len(failed_tests)} integration test(s) failed:\n{failure_summary}")
    if not ran_any_test:
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
    parser.add_argument("--test_suite", default="deepseek_v4", choices=sorted(_TEST_SUITES_FUNCTION))
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
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list, args.module, args.config)


if __name__ == "__main__":
    main()
