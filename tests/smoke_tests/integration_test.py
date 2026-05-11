# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/main/tests/integration_tests/run_tests.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test integration runner.

Launches a real end-to-end training flow (very few steps) via torchrun to
validate basic model and parallelism availability.

Usage:
    python tests/smoke_tests/integration_test.py ./outputs
    python tests/smoke_tests/integration_test.py ./outputs --test_name deepseek_v3_base
    python tests/smoke_tests/integration_test.py ./outputs --ngpu 4
"""

import argparse
import logging
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List  # noqa: PEA001

logger = logging.getLogger(__name__)

# ============================================================================
# Local OverrideDefinitions, kept aligned with upstream
# ============================================================================


@dataclass
class OverrideDefinitions:
    """Override definition for integration tests."""

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False

    def __repr__(self):
        return self.test_descr


# ============================================================================
# Runner
# ============================================================================

# NPU training entry module
_NPU_MODULE = "torchtitan_npu.models.deepseek_v3"
_NPU_CONFIG = "deepseek_v3_smoketest"
_NPU_TRAIN_FILE = "torchtitan_npu.entry"


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def _build_cmd(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    override_arg: Sequence[str],
) -> str:
    """Build the full training command."""
    test_name = test_flavor.test_name
    dump_folder = f"{output_dir}/{test_name}"
    dump_folder_arg = f"--dump_folder {dump_folder}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    cmd = (
        f"NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} " f"bash ./scripts/run_train.sh"
    )

    # Append dump_folder
    cmd += " " + dump_folder_arg

    # Append test override args
    if override_arg:
        cmd += " " + " ".join(override_arg)

    # Ensure dump directory exists
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    # Save logs
    log_path = f"{dump_folder}/test.log"
    cmd += f" 2>&1 | tee {log_path}"

    return cmd


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
):
    """Run one integration test."""
    test_name = test_flavor.test_name

    for override_arg in test_flavor.override_args:
        cmd = _build_cmd(test_flavor, output_dir, override_arg)

        logger.info(
            "===== %s Integration test, flavor: %s, command: %s =====",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            test_flavor.test_descr,
            cmd,
        )

        result = _run_cmd(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"\nFailed test flavor: {test_flavor.test_descr}.\n"
                f"Command: {cmd}\n"
                f"stderr: {result.stderr}\n"
            )


def run_tests(args, test_list: List[OverrideDefinitions]):
    """Run all integration tests."""
    ran_any_test = False
    failed_tests: list[tuple[str, str]] = []

    for test_flavor in test_list:
        # Filter by test_name
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue

        if test_flavor.disabled:
            continue

        # Check GPU count
        if args.ngpu < test_flavor.ngpu:
            logger.info(
                "Skipping test %s that requires %s gpus, because --ngpu arg is %s",
                test_flavor.test_name,
                test_flavor.ngpu,
                args.ngpu,
            )
        else:
            try:
                run_single_test(test_flavor, args.output_dir)
            except Exception as e:
                logger.error("ERROR: %s", e)
                failed_tests.append((test_flavor.test_name, str(e)))
            ran_any_test = True

    if failed_tests:
        failure_summary = "\n".join(
            f"  {name}: {error}" for name, error in failed_tests
        )
        raise RuntimeError(
            f"{len(failed_tests)} integration test(s) failed:\n{failure_summary}"
        )

    if not ran_any_test:
        available_tests = [t.test_name for t in test_list if not t.disabled]
        logger.warning(
            "No tests were run for --test_name '%s'. Available test names: %s",
            args.test_name,
            available_tests,
        )


# ============================================================================
# Test case definitions
# ============================================================================


def _base_tests() -> List[OverrideDefinitions]:
    """Base functionality test: DeepSeek V3 small-model training."""
    return [
        OverrideDefinitions(
            [
                [
                    f"--module {_NPU_MODULE}",
                    f"--config {_NPU_CONFIG}",
                ]
            ],
            "DeepSeek V3 BASE",
            "deepseek_v3_base",
            ngpu=2,
        ),
    ]


def _cp_tests() -> List[OverrideDefinitions]:
    """Context Parallel test."""
    return [
        OverrideDefinitions(
            [
                [
                    f"--module {_NPU_MODULE}",
                    f"--config {_NPU_CONFIG}",
                    "--parallelism.context_parallel_degree 2",
                    "--parallelism.enable_custom_context_parallel",
                ]
            ],
            "DeepSeek V3 CP Ulysses",
            "deepseek_v3_cp_ulysses",
            ngpu=2,
        ),
    ]


def _ep_tests() -> List[OverrideDefinitions]:
    """Expert Parallel test."""
    return [
        OverrideDefinitions(
            [
                [
                    f"--module {_NPU_MODULE}",
                    f"--config {_NPU_CONFIG}",
                    "--parallelism.expert_parallel_degree 2",
                ]
            ],
            "DeepSeek V3 FSDP+EP",
            "deepseek_v3_fsdp_ep",
            ngpu=2,
        ),
    ]


def generate_smoke_tests() -> List[OverrideDefinitions]:
    return _base_tests() + _ep_tests()


# ============================================================================
# Entry point
# ============================================================================


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="torchtitan-npu smoke tests")
    parser.add_argument("output_dir", help="Output directory for test results")
    parser.add_argument(
        "--test_name",
        default="all",
        help="Test name to run (for example, 'deepseek_v3_base'). Use 'all' to run every test.",
    )
    parser.add_argument(
        "--ngpu", default=2, type=int, help="Maximum available GPU count"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_list = generate_smoke_tests()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
