# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PARSER_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "training-log-visualization"
    / "scripts"
    / "train_log_plot.py"
)


def _load_parser_module():
    module_name = "training_log_visualization_test_parser"
    spec = importlib.util.spec_from_file_location(module_name, PARSER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PARSER = _load_parser_module()


def test_parser_supports_scientific_notation_and_nonfinite_values(tmp_path):
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "indexer loss: 6.8e-05\n"
        "step: 1 loss: 1.0e-04 grad_norm: -2.5E+03 "
        "memory: 1.25e1 GiB(5e1%) tps: inf "
        "elapsed_time_per_step: 2.5e-1s\n"
        "step: 2 loss: NaN grad_norm: +Infinity indexer loss: -inf\n"
        "step: 3 loss: inf grad_norm: -NaN\n",
        encoding="utf-8",
    )

    records, warnings = PARSER.read_training_metrics(log_path)

    assert warnings == []
    assert [record["step"] for record in records] == [1, 2, 3]
    assert records[0]["loss"] == pytest.approx(1.0e-4)
    assert records[0]["grad_norm"] == pytest.approx(-2.5e3)
    assert records[0]["memory_gib"] == pytest.approx(12.5)
    assert records[0]["memory_pct"] == pytest.approx(50.0)
    assert records[0]["indexer_loss"] == pytest.approx(6.8e-5)
    assert math.isinf(records[0]["tps"])
    assert records[0]["elapsed_time_per_step"] == pytest.approx(0.25)
    assert math.isnan(records[1]["loss"])
    assert records[1]["grad_norm"] == math.inf
    assert records[1]["indexer_loss"] == -math.inf
    assert records[2]["loss"] == math.inf
    assert math.isnan(records[2]["grad_norm"])


@pytest.mark.parametrize(
    "value",
    [
        "1.0e-04suffix",
        "1.0e",
        "6.8e-05.1",
        "1,23",
        "nanosecond",
        "infinite",
    ],
)
def test_numeric_regex_rejects_partial_tokens(tmp_path, value):
    log_path = tmp_path / "training.log"
    log_path.write_text(
        f"step: 1 loss: {value} grad_norm: 2.0\n",
        encoding="utf-8",
    )

    records, warnings = PARSER.read_training_metrics(log_path)

    assert records == []
    assert warnings == ["ignored 1 malformed metric lines"]


def test_reader_rejects_nonfinite_step_without_dropping_valid_records(tmp_path):
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "step: nan loss: 1.0 grad_norm: 2.0\nstep: 3 loss: 4.0 grad_norm: 5.0\n",
        encoding="utf-8",
    )

    records, warnings = PARSER.read_training_metrics(log_path)

    assert [record["step"] for record in records] == [3]
    assert warnings == ["ignored 1 malformed metric lines"]


def test_absolute_error_is_nonnegative_while_relative_error_keeps_direction():
    absolute_errors, relative_errors = PARSER.compute_errors(
        [2.0, -2.0],
        [1.0, -1.0],
        baseline="a",
    )

    assert absolute_errors == [1.0, 1.0]
    assert relative_errors == [-0.5, 0.5]
