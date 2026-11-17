# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Pure overlap metrics and candidate acceptance policy."""


def _overlap_metrics(coverage, *, cube_work, vector_work, compute_union=None, span=None):
    cv, mixed_cv, cc, vv = coverage
    cube_busy = max(0.0, cube_work - cc)
    vector_busy = max(0.0, vector_work - vv)
    result = {
        "cv_ms": cv,
        "mixed_cv_ms": mixed_cv,
        "cc_ms": cc,
        "vv_ms": vv,
        "cube_busy_ms": cube_busy,
        "vector_busy_ms": vector_busy,
        "cube_covered_by_vector_percent": 100 * cv / cube_busy if cube_busy else 0.0,
        "vector_covered_by_cube_percent": 100 * cv / vector_busy if vector_busy else 0.0,
    }
    if compute_union is not None:
        result["compute_union_ms"] = compute_union
        result["cv_percent_of_compute"] = 100 * cv / compute_union if compute_union else 0.0
    if span is not None:
        result["span_ms"] = span
    return result


def _local_coverage_score(coverage):
    cv, mixed_cv, cc, vv = coverage
    return (round(cv, 9), -round(cc + vv, 9), -round(cc, 9), -round(vv, 9), round(mixed_cv, 9))


def _metric_score(metrics):
    cube = metrics["cube_covered_by_vector_percent"]
    vector = metrics["vector_covered_by_cube_percent"]
    return (
        round(min(cube, vector), 9),
        round(cube + vector, 9),
        round(metrics["cv_ms"], 9),
        -round(metrics["cc_ms"] + metrics["vv_ms"], 9),
        -round(metrics["cc_ms"], 9),
        -round(metrics["vv_ms"], 9),
        round(metrics["mixed_cv_ms"], 9),
        -round(metrics["span_ms"], 9),
    )


def candidate_score(current, current_score, metrics):
    current_cube, current_vector, current_cc, current_vv = current
    cube = metrics["cube_covered_by_vector_percent"]
    vector = metrics["vector_covered_by_cube_percent"]
    cc = metrics["cc_ms"]
    vv = metrics["vv_ms"]
    non_regression = (
        cube + 1e-9 >= current_cube
        and vector + 1e-9 >= current_vector
        and cc <= current_cc + 1e-9
        and vv <= current_vv + 1e-9
    )
    score = _metric_score(metrics)
    if not non_regression or score <= current_score:
        return None
    return score


def predicted_coverage_improved(free_prediction, scheduled_prediction):
    """Require strict bidirectional gains for the final baseline comparison."""
    return (
        round(
            scheduled_prediction["cube_covered_by_vector_percent"] - free_prediction["cube_covered_by_vector_percent"],
            9,
        )
        > 0
        and round(
            scheduled_prediction["vector_covered_by_cube_percent"] - free_prediction["vector_covered_by_cube_percent"],
            9,
        )
        > 0
        and scheduled_prediction["cc_ms"] <= free_prediction["cc_ms"] + 1e-9
        and scheduled_prediction["vv_ms"] <= free_prediction["vv_ms"] + 1e-9
    )
