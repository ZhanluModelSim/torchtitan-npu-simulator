# Copyright (c) 2026 Ethan_Zou. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host regressions for numerics log comparison."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling
SCRIPT = Path(__file__).resolve().parents[4] / ".agents/skills/numerics-debugging/scripts/compare_numerics.py"


@pytest.fixture
def compare(monkeypatch):
    spec = importlib.util.spec_from_file_location("numerics_host_regression", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def log_entry(compare, tmp_path, filename, *, key, phase="forward", output="1.0", inputs="2.0"):
    path = tmp_path / filename
    path.write_text(
        f"[{key}]\n  Shape: torch.Size([1])\n  Output hash: {output}\n"
        f"  Input hashes: {inputs}\n  Phase: {phase}\n", encoding="utf-8"
    )
    entries, _ = compare.parse_log(str(path))
    return entries


@pytest.mark.parametrize("eager_key,traced_key", [
    ("layers.0/op_0_mul", "layers.0/op_0_mul"),
    ("layers.0/op_0_mul", "layers.0/op_1_mul"),
    ("layers.0/op_0_mul", "layers.1/op_1_mul"),
], ids=["exact", "fuzzy", "stats"])
def test_automatic_matching_keeps_forward_and_backward_separate(compare, tmp_path, eager_key, traced_key):
    eager = log_entry(compare, tmp_path, "eager.log", key=eager_key, phase="forward")
    traced = log_entry(compare, tmp_path, "traced.log", key=traced_key, phase="backward")

    results = compare.match_entries(eager, traced)

    assert [row[2] for row in results] == ["eager_only", "traced_only"]
    assert results[0][0].phase == "forward"
    assert results[1][1].phase == "backward"


@pytest.mark.parametrize("eager_key,traced_key,strategy", [
    ("layers.0/op_0_mul", "layers.0/op_0_mul", "exact"),
    ("layers.0/op_0_mul", "layers.0/op_1_mul", "fuzzy"),
    ("layers.0/op_0_mul", "layers.1/op_1_mul", "stats"),
], ids=["exact", "fuzzy", "stats"])
def test_automatic_matching_preserves_same_phase_pairs(compare, tmp_path, eager_key, traced_key, strategy):
    eager = log_entry(compare, tmp_path, "eager.log", key=eager_key, phase="backward")
    traced = log_entry(compare, tmp_path, "traced.log", key=traced_key, phase="backward")

    results = compare.match_entries(eager, traced)

    assert len(results) == 1
    assert results[0][2] == "match"
    assert results[0][4] == strategy


def test_explicit_override_can_still_force_a_cross_phase_pair(compare, tmp_path):
    eager_key, traced_key = "layers.0/op_0_mul", "layers.1/op_1_mul"
    eager = log_entry(compare, tmp_path, "eager.log", key=eager_key, phase="forward")
    traced = log_entry(compare, tmp_path, "traced.log", key=traced_key, phase="backward")

    results = compare.match_entries(eager, traced, overrides={eager_key: traced_key})

    assert len(results) == 1
    assert results[0][4] == "override"
