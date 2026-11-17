# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Hand-computed candidate policy boundaries."""

import pytest

from torchtitan_npu.extensions.experiment.cv_parallel.schedule_scoring import (
    candidate_score,
    predicted_coverage_improved,
)


@pytest.mark.parametrize(
    "vector,cc,accepted",
    [(51.0, 0.0, True), (50.0, 0.0, False), (49.0, 0.0, False), (51.0, 2e-9, False)],
)
def test_candidate_policy_preserves_directional_and_same_type_guards(vector, cc, accepted):
    metrics = {
        "cube_covered_by_vector_percent": 50.0,
        "vector_covered_by_cube_percent": vector,
        "cv_ms": 5.0,
        "mixed_cv_ms": 0.0,
        "cc_ms": cc,
        "vv_ms": 0.0,
        "span_ms": 10.0,
    }
    # Baseline: 50% in both directions, 5 ms CV, zero same-type overlap.
    baseline_score = (50.0, 100.0, 5.0, 0.0, 0.0, 0.0, 0.0, -10.0)
    result = candidate_score((50.0, 50.0, 0.0, 0.0), baseline_score, metrics)
    assert (result is not None) is accepted
    baseline = {**metrics, "vector_covered_by_cube_percent": 50.0, "cc_ms": 0.0}
    # Final validation requires both directions to improve, unlike a local edit.
    assert not predicted_coverage_improved(baseline, metrics)


def test_final_prediction_requires_strict_gains_after_rounding():
    baseline = {
        "cube_covered_by_vector_percent": 50.0,
        "vector_covered_by_cube_percent": 50.0,
        "cc_ms": 0.0,
        "vv_ms": 0.0,
    }
    assert predicted_coverage_improved(
        baseline,
        {**baseline, "cube_covered_by_vector_percent": 51.0, "vector_covered_by_cube_percent": 51.0},
    )
    assert not predicted_coverage_improved(
        baseline,
        {**baseline, "cube_covered_by_vector_percent": 50.0 + 1e-10, "vector_covered_by_cube_percent": 51.0},
    )
