# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script_module():  # noqa: ANN202
    script_path = Path(__file__).parents[3] / "scripts" / "run_simulator_spawn.py"
    spec = importlib.util.spec_from_file_location("run_simulator_spawn", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_script_module()


def _process(exitcode: int | None = 0) -> SimpleNamespace:
    return SimpleNamespace(exitcode=exitcode)


def _result(rank: int, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir()
    return {
        "rank": rank,
        "workload_id": f"workload-{rank}",
        "step_template_count": 3,
        "step_template_ids": [f"s{rank}_F", f"s{rank}_B", f"s{rank}_OPTIMIZER"],
        "output_dir": str(output_dir),
    }


def test_validate_worker_results_accepts_one_complete_result_per_rank(tmp_path: Path) -> None:
    results = [
        _result(1, tmp_path / "rank_1"),
        _result(0, tmp_path / "rank_0"),
    ]

    validated = launcher._validate_worker_results(
        results, [_process(), _process()], expected_count=2
    )

    assert [result["rank"] for result in validated] == [0, 1]


def test_validate_worker_results_rejects_failed_process(tmp_path: Path) -> None:
    results = [_result(0, tmp_path / "rank_0")]

    with pytest.raises(RuntimeError, match=r"rank 1 \(exitcode 7\)"):
        launcher._validate_worker_results(
            results, [_process(), _process(7)], expected_count=2
        )


@pytest.mark.parametrize(
    ("ranks", "error"),
    [
        ([0], "missing=\\[1\\]"),
        ([0, 0], "duplicate=\\[0\\]"),
        ([0, 2], "unexpected=\\[2\\]"),
    ],
)
def test_validate_worker_results_rejects_incomplete_rank_sets(
    tmp_path: Path, ranks: list[int], error: str
) -> None:
    results = [
        _result(rank, tmp_path / f"result_{index}")
        for index, rank in enumerate(ranks)
    ]

    with pytest.raises(RuntimeError, match=error):
        launcher._validate_worker_results(
            results, [_process(), _process()], expected_count=2
        )


def test_validate_worker_results_rejects_empty_capture(tmp_path: Path) -> None:
    result = _result(0, tmp_path / "rank_0")
    result["step_template_count"] = 0

    with pytest.raises(RuntimeError, match="captured no step templates"):
        launcher._validate_worker_results([result], [_process()], expected_count=1)


def test_validate_worker_results_rejects_missing_stage_capture(tmp_path: Path) -> None:
    result = _result(1, tmp_path / "rank_1")
    result["step_template_ids"] = ["s-1_F", "s1_F", "s1_OPTIMIZER"]

    with pytest.raises(RuntimeError, match=r"missing step templates \['s1_B'\]"):
        launcher._validate_worker_results(
            [
                _result(0, tmp_path / "rank_0"),
                result,
            ],
            [_process(), _process()],
            expected_count=2,
        )


def test_validate_worker_results_rejects_missing_output_directory(tmp_path: Path) -> None:
    result = {
        "rank": 0,
        "workload_id": "workload-0",
        "step_template_count": 3,
        "step_template_ids": ["s0_F", "s0_B", "s0_OPTIMIZER"],
        "output_dir": str(tmp_path / "missing"),
    }

    with pytest.raises(RuntimeError, match="output directory is missing"):
        launcher._validate_worker_results([result], [_process()], expected_count=1)


def test_collect_worker_results_stops_when_a_process_fails() -> None:
    with pytest.raises(RuntimeError, match=r"rank 1 \(exitcode 3\)"):
        launcher._collect_worker_results(
            queue.Queue(), [_process(), _process(3)], expected_count=2, timeout=0.1
        )
