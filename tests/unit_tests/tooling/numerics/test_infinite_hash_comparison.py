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


@pytest.mark.parametrize("field", ["output", "inputs"], ids=["output_hash", "input_hashes"])
@pytest.mark.parametrize("first,second", [("inf", "1.0"), ("1.0", "inf"), ("inf", "-inf")],
                         ids=["infinite_finite", "finite_infinite", "opposite_infinities"])
def test_hash_comparison_marks_different_infinities_as_diff(compare, tmp_path, field, first, second):
    eager = log_entry(compare, tmp_path, "eager.log", key="layer/op_0_mul", **{field: first})
    traced = log_entry(compare, tmp_path, "traced.log", key="layer/op_0_mul", **{field: second})

    results = compare.match_entries(eager, traced)

    assert results[0][2] == "diff"


@pytest.mark.parametrize("first,second", [("inf", "+Infinity"), ("-inf", "-Infinity"), ("nan", "nan")],
                         ids=["positive_infinity", "negative_infinity", "unsynced_nan"])
def test_hash_comparison_preserves_equivalent_nonfinite_tokens(compare, tmp_path, first, second):
    eager = log_entry(compare, tmp_path, "eager.log", key="layer/op_0_mul", output=first)
    traced = log_entry(compare, tmp_path, "traced.log", key="layer/op_0_mul", output=second)

    results = compare.match_entries(eager, traced)

    assert results[0][2] == "match"
