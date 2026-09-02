#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/cc286a63599e42480a07928cc362e514ae448a85/scripts/loss_compare.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recorded-loss system copied from Torchtitan's ``scripts/loss_compare.py``."""

from __future__ import annotations

import os
import unittest
from pathlib import Path


LOG_PREFIX = "[LOSS_COMPARE]"
TB_LOSS_TAG = "loss_metrics/global_avg_loss"


def log_print(message: str = "") -> None:
    if message:
        print(f"{LOG_PREFIX} {message}")
    else:
        print(LOG_PREFIX)


def extract_losses_from_tensorboard(
    job_dump_folder: str | Path, tb_folder: str
) -> dict[int, float]:
    """Copied from Torchtitan's loss comparison runner."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    base_path = os.path.join(str(job_dump_folder), tb_folder)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"TensorBoard path does not exist: {base_path}")

    subdirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly one subdirectory under {base_path}, "
            f"found {len(subdirs)}: {subdirs}"
        )

    event_dir = os.path.join(base_path, subdirs[0])
    log_print(f"Loading TensorBoard events from: {event_dir}")
    event_acc = EventAccumulator(event_dir)
    event_acc.Reload()
    available_tags = event_acc.Tags().get("scalars", [])
    if TB_LOSS_TAG not in available_tags:
        raise KeyError(
            f"Scalar tag '{TB_LOSS_TAG}' not found in TensorBoard events. "
            f"Available tags: {available_tags}"
        )
    losses = {
        scalar.step: scalar.value for scalar in event_acc.Scalars(TB_LOSS_TAG)
    }
    log_print(f"Extracted {len(losses)} steps from TensorBoard events")
    return losses


def read_losses_from_file(loss_file: str | Path) -> dict[int, float]:
    """Copied from Torchtitan's loss comparison runner."""

    losses = {}
    with open(loss_file, encoding="utf-8") as f:
        for line in f:
            step, loss = line.strip().split()
            losses[int(step)] = float(loss)
    return losses


def assert_losses_equal(
    baseline_losses: dict[int, float],
    test_losses: dict[int, float] | None = None,
    import_result: str | Path | None = None,
) -> None:
    """Copied from Torchtitan, with explicit missing/extra step diagnostics."""

    assert baseline_losses, "No losses found in golden data"
    if test_losses is not None:
        assert test_losses, "No losses found in test TensorBoard data"
    imported_losses = read_losses_from_file(import_result) if import_result else None

    class LossEqualityTest(unittest.TestCase):
        def test_losses_equal(self):
            baseline_steps = set(baseline_losses.keys())
            if test_losses is not None:
                test_steps = set(test_losses.keys())
                self.assertEqual(
                    baseline_steps,
                    test_steps,
                    "Steps mismatch: "
                    f"missing_in_test={sorted(baseline_steps - test_steps)}, "
                    f"extra_in_test={sorted(test_steps - baseline_steps)}",
                )
            if imported_losses:
                imported_steps = set(imported_losses.keys())
                self.assertEqual(
                    baseline_steps,
                    imported_steps,
                    "Steps mismatch: "
                    f"missing_in_import={sorted(baseline_steps - imported_steps)}, "
                    f"extra_in_import={sorted(imported_steps - baseline_steps)}",
                )

            for step in sorted(baseline_steps):
                if test_losses is not None:
                    self.assertEqual(
                        baseline_losses[step],
                        test_losses[step],
                        f"Loss mismatch at step {step}: "
                        f"baseline={repr(baseline_losses[step])}, "
                        f"test={repr(test_losses[step])}",
                    )
                if imported_losses:
                    self.assertEqual(
                        baseline_losses[step],
                        imported_losses[step],
                        f"Loss mismatch at step {step}: "
                        f"baseline={repr(baseline_losses[step])}, "
                        f"imported={repr(imported_losses[step])}",
                    )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(LossEqualityTest)
    )
    if not result.wasSuccessful():
        raise AssertionError("Loss assertion failed!")
