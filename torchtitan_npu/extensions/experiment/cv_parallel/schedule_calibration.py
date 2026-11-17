# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Calibrate free execution and report scheduled CV coverage without measured rollback."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torchtitan.tools.logging import logger

from .cv_parallel import DualStreamRuntime, dependency_only_schedule, materialize
from .schedule_search import build_profile_schedule as build_local_profile_schedule
from .whole_graph_runtime_profile import collective_order_keys, node_id, profile_whole_graph_costs

if TYPE_CHECKING:
    from pathlib import Path


def gather_rank_values(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    # None is a valid payload, including successful alignment validation.
    uncollected = object()
    gathered: list[Any] = [uncollected] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if any(item is uncollected for item in gathered):
        raise RuntimeError("Failed to collect every rank's schedule measurements")
    return gathered


def build_profile_schedule(gm, profile, *, kernel_observations=None, reference=None):
    """Validate rank-local plans before any candidate graph executes."""
    schedule, stats = build_local_profile_schedule(
        gm, profile, kernel_observations=kernel_observations, reference=reference
    )
    keys = collective_order_keys(gm)
    communication = [keys[n] for n in schedule.order if n in keys]
    signature = hashlib.sha256(repr(communication).encode()).hexdigest()
    validation = gather_rank_values(
        {
            "signature": signature,
            "changed": stats.pop("changed_from_free"),
            "coverage_improved": stats.pop("predicted_coverage_improved"),
            "prediction": stats.pop("prediction"),
        }
    )
    if any(value["signature"] != signature for value in validation):
        raise RuntimeError("Collective order differs across ranks; refusing inconsistent events")
    stats.update(
        {
            "changed_from_free_by_rank": [v["changed"] for v in validation],
            "predicted_coverage_improved_by_rank": [v["coverage_improved"] for v in validation],
            # Preserve the existing report alias for bidirectional CV gains.
            "predicted_cv_improved_by_rank": [v["coverage_improved"] for v in validation],
            "prediction_by_rank": [v["prediction"] for v in validation],
        }
    )
    return schedule, stats


def _measure_schedule(graph, run_candidate, *, device, warmup, samples, label):
    """Synchronized F/B only; state copies, measurement collectives and optimizer excluded."""
    backend = torch.get_device_module(device)
    distributed = dist.is_available() and dist.is_initialized()
    main_stream = backend.current_stream(device)
    measurements = []
    elapsed = torch.empty(1, device=device, dtype=torch.float32)
    backend.synchronize(device)
    backend.empty_cache()
    backend.reset_peak_memory_stats(device)
    with torch.random.fork_rng(devices=[device.index], device_type=device.type):
        cpu_rng = torch.random.get_rng_state()
        device_rng = backend.get_rng_state(device)
        for iteration in range(warmup + samples):
            torch.random.set_rng_state(cpu_rng)
            backend.set_rng_state(device_rng, device)
            run_candidate.prepare()
            try:
                if distributed:
                    dist.barrier()
                backend.synchronize(device)
                started = time.perf_counter()
                result = run_candidate(graph)
                backend.synchronize(device)
                local_ms = (time.perf_counter() - started) * 1000
                del result
            finally:
                try:
                    backend.synchronize(device)
                finally:
                    backend.set_stream(main_stream)
                    run_candidate.finalize()
            elapsed.fill_(local_ms)
            if distributed:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            max_rank_ms = elapsed.item()
            if iteration >= warmup:
                measurements.append(max_rank_ms)
            logger.info(
                "CV parallel replay: algorithm=%s warmup=%s max_rank_ms=%.3f", label, iteration < warmup, max_rank_ms
            )
    return {
        "samples_ms": measurements,
        "median_ms": statistics.median(measurements),
        "peak_allocated_bytes_by_rank": gather_rank_values(backend.max_memory_allocated(device)),
        "peak_reserved_bytes_by_rank": gather_rank_values(backend.max_memory_reserved(device)),
    }


def _kernel_overlap(summaries):
    times = [summary["time_ms"] for summary in summaries]
    cv = sum(values["cv_overlap"] for values in times)
    cube_mix_aiv = sum(values["cube_mix_aiv_overlap"] for values in times)
    vector_mix_aic = sum(values["vector_mix_aic_overlap"] for values in times)
    cc = sum(values["cc_overlap"] for values in times)
    vv = sum(values["vv_overlap"] for values in times)
    union = sum(values["compute_union"] for values in times)
    if union <= 0:
        raise ValueError("No compute kernels in the forward/backward profile")
    return {
        "cc_max_ms": max(values["cc_overlap"] for values in times),
        "vv_max_ms": max(values["vv_overlap"] for values in times),
        "cc_mean_ms": cc / len(times),
        "vv_mean_ms": vv / len(times),
        "cv_mean_ms": cv / len(times),
        "cv_percent_of_compute": 100 * cv / union,
        "cube_mix_aiv_mean_ms": cube_mix_aiv / len(times),
        "vector_mix_aic_mean_ms": vector_mix_aic / len(times),
        "cv_with_cube_mix_aiv_mean_ms": (cv + cube_mix_aiv) / len(times),
        "cv_with_cube_mix_aiv_percent_of_compute": 100 * (cv + cube_mix_aiv) / union,
        "cube_covered_by_vector_mean_percent": statistics.mean(
            100 * values["cv_overlap"] / values["cube_busy"] if values["cube_busy"] else 0.0 for values in times
        ),
        "vector_covered_by_cube_mean_percent": statistics.mean(
            100 * values["cv_overlap"] / values["vector_busy"] if values["vector_busy"] else 0.0 for values in times
        ),
    }


def _prediction_summary(predictions):
    result: dict[str, Any] = {}
    for mode in ("free", "scheduled"):
        rows = [prediction[mode] for prediction in predictions]
        cv = statistics.mean(row["cv_ms"] for row in rows)
        union = statistics.mean(row["compute_union_ms"] for row in rows)
        result[mode] = {
            "cv_mean_ms": cv,
            "compute_union_mean_ms": union,
            "cv_percent_of_compute": 100 * cv / union if union else None,
            "cube_covered_by_vector_mean_percent": statistics.mean(
                row["cube_covered_by_vector_percent"] for row in rows
            ),
            "vector_covered_by_cube_mean_percent": statistics.mean(
                row["vector_covered_by_cube_percent"] for row in rows
            ),
            "cc_mean_ms": statistics.mean(row["cc_ms"] for row in rows),
            "vv_mean_ms": statistics.mean(row["vv_ms"] for row in rows),
            "span_max_ms": max(row["span_ms"] for row in rows),
        }
    free, scheduled = result["free"], result["scheduled"]
    gain = scheduled["cv_mean_ms"] - free["cv_mean_ms"]
    result["gain"] = {
        "cv_mean_ms": gain,
        "cv_relative_percent": 100 * gain / free["cv_mean_ms"] if free["cv_mean_ms"] else None,
        "cv_share_percentage_points": scheduled["cv_percent_of_compute"] - free["cv_percent_of_compute"]
        if free["cv_percent_of_compute"] is not None and scheduled["cv_percent_of_compute"] is not None
        else None,
        "span_reduction_ms": free["span_max_ms"] - scheduled["span_max_ms"],
        "cube_covered_by_vector_delta_pp": scheduled["cube_covered_by_vector_mean_percent"]
        - free["cube_covered_by_vector_mean_percent"],
        "vector_covered_by_cube_delta_pp": scheduled["vector_covered_by_cube_mean_percent"]
        - free["vector_covered_by_cube_mean_percent"],
        "cc_reduction_ms": free["cc_mean_ms"] - scheduled["cc_mean_ms"],
        "vv_reduction_ms": free["vv_mean_ms"] - scheduled["vv_mean_ms"],
    }
    return result


def _validate_cv_overlap(free_summaries, summaries):
    rows = []
    for free_rank, scheduled_rank in zip(free_summaries, summaries, strict=True):
        free, scheduled = free_rank["time_ms"], scheduled_rank["time_ms"]
        free_ms, scheduled_ms = free["cv_overlap"], scheduled["cv_overlap"]
        free_union, scheduled_union = free["compute_union"], scheduled["compute_union"]
        free_ns, scheduled_ns = round(free_ms * 1e6), round(scheduled_ms * 1e6)
        free_union_ns, scheduled_union_ns = round(free_union * 1e6), round(scheduled_union * 1e6)
        if free_union_ns <= 0 or scheduled_union_ns <= 0:
            raise ValueError("No compute kernels in the forward/backward profile")
        time_passed = scheduled_ns > free_ns
        share_passed = scheduled_ns * free_union_ns > free_ns * scheduled_union_ns
        free_percent, scheduled_percent = 100 * free_ms / free_union, 100 * scheduled_ms / scheduled_union
        free_cube_percent = 100 * free_ms / free["cube_busy"] if free["cube_busy"] else 0.0
        scheduled_cube_percent = 100 * scheduled_ms / scheduled["cube_busy"] if scheduled["cube_busy"] else 0.0
        free_vector_percent = 100 * free_ms / free["vector_busy"] if free["vector_busy"] else 0.0
        scheduled_vector_percent = 100 * scheduled_ms / scheduled["vector_busy"] if scheduled["vector_busy"] else 0.0
        cube_coverage_passed = round(scheduled_cube_percent - free_cube_percent, 9) > 0
        vector_coverage_passed = round(scheduled_vector_percent - free_vector_percent, 9) > 0
        cc_passed = round(scheduled["cc_overlap"] - free["cc_overlap"], 9) <= 0
        vv_passed = round(scheduled["vv_overlap"] - free["vv_overlap"], 9) <= 0
        rows.append(
            {
                "free_ms": free_ms,
                "scheduled_ms": scheduled_ms,
                "delta_ms": scheduled_ms - free_ms,
                "free_percent_of_compute": free_percent,
                "scheduled_percent_of_compute": scheduled_percent,
                "share_delta_percentage_points": scheduled_percent - free_percent,
                "free_cube_covered_by_vector_percent": free_cube_percent,
                "scheduled_cube_covered_by_vector_percent": scheduled_cube_percent,
                "cube_covered_by_vector_delta_percentage_points": scheduled_cube_percent - free_cube_percent,
                "free_vector_covered_by_cube_percent": free_vector_percent,
                "scheduled_vector_covered_by_cube_percent": scheduled_vector_percent,
                "vector_covered_by_cube_delta_percentage_points": scheduled_vector_percent - free_vector_percent,
                "cc_delta_ms": scheduled["cc_overlap"] - free["cc_overlap"],
                "vv_delta_ms": scheduled["vv_overlap"] - free["vv_overlap"],
                "time_passed": time_passed,
                "share_passed": share_passed,
                "cube_coverage_passed": cube_coverage_passed,
                "vector_coverage_passed": vector_coverage_passed,
                "cc_passed": cc_passed,
                "vv_passed": vv_passed,
                "passed": time_passed
                and share_passed
                and cube_coverage_passed
                and vector_coverage_passed
                and cc_passed
                and vv_passed,
            }
        )
    return rows


def _final_optimization_summary(planning_stats, *, applied):
    if applied and planning_stats.get("optimization_summary"):
        return planning_stats["optimization_summary"]
    zero_benefit = {
        "cv_gain_ms": 0.0,
        "mixed_cv_gain_ms": 0.0,
        "cc_reduction_ms": 0.0,
        "vv_reduction_ms": 0.0,
        "span_reduction_ms": 0.0,
        "cube_coverage_gain_percentage_points": 0.0,
        "vector_coverage_gain_percentage_points": 0.0,
    }
    return {
        "attribution_method": "not_applied",
        "reorder": {"actions": 0, "reordered_nodes": 0, "estimated_benefit": zero_benefit.copy()},
        "wait_event": {
            "accepted_alignment_edits": 0,
            "alignment_gates_kept": 0,
            "waits_added": 0,
            "waits_removed": 0,
            "waits_retargeted": 0,
            "event_records_total": 0,
            "estimated_benefit": zero_benefit.copy(),
        },
        "total_estimated_benefit": zero_benefit.copy(),
    }


def _log_final_optimization(selected, summary):
    reorder = summary["reorder"]
    wait = summary["wait_event"]
    logger.info(
        "CV parallel final optimization: selected=%s reordered_nodes=%d reorder_actions=%d "
        "wait_events_added=%d wait_events_removed=%d wait_events_retargeted=%d alignment_gates=%d "
        "estimated_reorder_benefit=%s estimated_wait_event_benefit=%s",
        selected,
        reorder["reordered_nodes"],
        reorder["actions"],
        wait["waits_added"],
        wait["waits_removed"],
        wait["waits_retargeted"],
        wait["alignment_gates_kept"],
        reorder["estimated_benefit"],
        wait["estimated_benefit"],
    )


def calibrate_and_select_schedule(gm, run_candidate, *, profile_root: Path):
    """Install a predicted candidate and report its native measurements on every rank."""
    samples, warmup = 3, 1
    started = time.perf_counter()
    device = torch.device("npu", torch.npu.current_device())  # pyrefly: ignore [missing-attribute]
    backend = torch.get_device_module(device)
    runtime = DualStreamRuntime()
    free_plan = dependency_only_schedule(gm)
    free, _ = materialize(gm, free_plan, runtime=runtime)
    calibration_warmup = 3
    logger.info("CV parallel calibration: current dependency_only, same batch and RNG; balanced CV with CC/VV guards")
    free_measurement = _measure_schedule(
        free,
        run_candidate,
        device=device,
        warmup=calibration_warmup,
        samples=samples,
        label="dependency_only_calibration",
    )
    free_summary, kernel_observations = {}, {}
    profile = profile_whole_graph_costs(
        free,
        run_candidate,
        profile_name="dependency_only",
        profile_root=profile_root,
        source_gm=gm,
        kernel_summary=free_summary,
        kernel_observations=kernel_observations,
    )
    free_summaries = gather_rank_values(free_summary)
    free_overlap = _kernel_overlap(free_summaries)
    schedule, planning_stats = build_profile_schedule(
        gm,
        profile,
        kernel_observations=kernel_observations,
        reference=free_plan,
    )
    output = profile_root / "schedule_feedback.json"

    def save(report):
        if not dist.is_initialized() or dist.get_rank() == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")

    report: dict[str, Any] = {
        "status": "planned",
        "selected": None,
        "selection": schedule.algorithm,
        "cc_policy": "non_increasing",
        "vv_policy": "non_increasing; no_vv_search",
        "scope": "synchronized_forward_backward_without_profiler_or_optimizer",
        "rank_aggregation": "maximum_per_replay",
        "calibration": "same_batch_dependency_only",
        "profile_reuse": "none; current device each startup",
        "hardware": {"device_name": backend.get_device_name(device), "torch_version": torch.__version__},
        "free_profile_overlap": free_overlap,
        "free_profile_overlap_by_rank": free_summaries,
        "free_calibration_measurement": free_measurement,
        "warmup_runs": warmup,
        "calibration_warmup_runs": calibration_warmup,
        "measured_runs": samples,
        "data_batches_consumed": 0,
        "schedule": {"planner": planning_stats},
        "prediction": _prediction_summary(planning_stats["prediction_by_rank"]),
        "planned_optimization": planning_stats.get("optimization_summary"),
    }
    logger.info("CV parallel predicted gains: %s", json.dumps(report["prediction"], sort_keys=True))
    windows = planning_stats["window_refinement"]
    logger.info(
        "CV parallel schedule: algorithm=%s predicted_cv_ms=%.3f->%.3f "
        "predicted_cc_ms=%.3f->%.3f predicted_vv_ms=%.3f->%.3f "
        "cube_coverage=%.3f->%.3f vector_coverage=%.3f->%.3f "
        "span_ms=%.3f->%.3f planning_seconds=%.3f search_seconds=%.3f",
        schedule.algorithm,
        planning_stats["predicted_free_cv_ms"],
        planning_stats["predicted_cv_ms"],
        planning_stats["predicted_free_cc_ms"],
        planning_stats["predicted_cc_ms"],
        planning_stats["predicted_free_vv_ms"],
        planning_stats["predicted_vv_ms"],
        planning_stats["predicted_free_cube_covered_by_vector_percent"],
        planning_stats["predicted_cube_covered_by_vector_percent"],
        planning_stats["predicted_free_vector_covered_by_cube_percent"],
        planning_stats["predicted_vector_covered_by_cube_percent"],
        planning_stats["predicted_free_span_ms"],
        planning_stats["predicted_span_ms"],
        planning_stats["planning_wall_seconds"],
        planning_stats["search_wall_seconds"],
    )
    logger.info(
        "CV parallel wait cleanup: gates_pruned=%d; search counters and per-node coverage: %s",
        windows.get("num_gates_pruned", 0),
        output,
    )
    logger.info(
        "CV parallel waits: free=%d scheduled=%d added=%d removed=%d retargeted=%d; cube_exclusion_edges=0",
        planning_stats["num_waits_free"],
        planning_stats["num_waits"],
        planning_stats["num_waits_added"],
        planning_stats["num_waits_removed"],
        planning_stats["num_waits_retargeted"],
    )
    if not all(planning_stats["changed_from_free_by_rank"]) or not all(
        planning_stats["predicted_coverage_improved_by_rank"]
    ):
        applied_optimization = _final_optimization_summary(planning_stats, applied=False)
        report.update(
            status="fallback",
            selected="dependency_only",
            fallback_reason="no_effective_predicted_cv_gain",
            applied_optimization=applied_optimization,
            profile_graph_replays=1,
            total_graph_replays=calibration_warmup + samples + 1,
            feedback_wall_seconds=time.perf_counter() - started,
        )
        save(report)
        _log_final_optimization("dependency_only", applied_optimization)
        logger.warning("No CV-improving schedule on every rank; using dependency_only. Report: %s", output)
        return free
    save(report)
    del free
    rank = dist.get_rank() if dist.is_initialized() else 0
    runtime.synchronize_for_topology_rebind()
    candidate, _ = materialize(gm, schedule, runtime=runtime)
    measurement = _measure_schedule(
        candidate,
        run_candidate,
        device=device,
        warmup=warmup,
        samples=samples,
        label=f"{schedule.algorithm}_validation",
    )
    summary = {}
    scheduled_observations = {}
    scheduled_profile = profile_whole_graph_costs(
        candidate,
        run_candidate,
        profile_name=f"{schedule.algorithm}_validation",
        profile_root=profile_root,
        source_gm=gm,
        kernel_summary=summary,
        kernel_observations=scheduled_observations,
    )
    node_profiles = profile_root / f"node_profiles_rank{rank}.json"
    node_profiles.write_text(
        json.dumps(
            {
                "nodes": {
                    node_id(n): {"name": n.name, "target": str(n.target), "chunk_id": n.meta.get("chunk_id")}
                    for n in gm.graph.nodes
                },
                "free": {
                    "local_ms": profile.local_ms,
                    "local_resources": profile.local_resources,
                    "kernel_observations": kernel_observations,
                },
                "scheduled": {
                    "local_ms": scheduled_profile.local_ms,
                    "local_resources": scheduled_profile.local_resources,
                    "kernel_observations": scheduled_observations,
                },
            }
        )
        + "\n"
    )
    node_profile_files = gather_rank_values(str(node_profiles))
    summaries = gather_rank_values(summary)
    cv_validation = _validate_cv_overlap(free_summaries, summaries)
    overlap = _kernel_overlap(summaries)
    delta = {key: overlap[key] - free_overlap[key] for key in overlap}
    cv_verified = all(item["time_passed"] and item["share_passed"] for item in cv_validation)
    directional_coverage_verified = all(
        item["cube_coverage_passed"] and item["vector_coverage_passed"] for item in cv_validation
    )
    same_type_verified = all(item["cc_passed"] and item["vv_passed"] for item in cv_validation)
    elapsed_delta = measurement["median_ms"] - free_measurement["median_ms"]
    elapsed_verified = elapsed_delta <= 0
    validation_passed = cv_verified and directional_coverage_verified and same_type_verified and elapsed_verified
    validation_warning = None
    if not cv_verified:
        validation_warning = "measured_cv_not_improved"
    elif not directional_coverage_verified:
        validation_warning = "measured_directional_coverage_not_improved"
    elif not same_type_verified:
        validation_warning = "measured_same_type_overlap_increased"
    elif not elapsed_verified:
        validation_warning = "forward_backward_regression"

    feedback = {
        "planner": planning_stats,
        **measurement,
        "kernel_overlap_by_rank": summaries,
        "kernel_overlap": overlap,
        "delta_from_free_profile": delta,
        "forward_backward_delta_ms": elapsed_delta,
        "forward_backward_guard_passed": elapsed_verified,
        "measured_cv_increase_verified": cv_verified,
        "measured_directional_coverage_verified": directional_coverage_verified,
        "measured_same_type_overlap_non_increase_verified": same_type_verified,
        "measured_cv_share_increase_verified": all(item["share_passed"] for item in cv_validation),
        "validation_passed": validation_passed,
        "validation_warning": validation_warning,
        "cv_validation_by_rank": cv_validation,
    }
    del scheduled_profile, scheduled_observations, profile, kernel_observations
    applied_optimization = _final_optimization_summary(planning_stats, applied=True)
    report.update(
        status="applied",
        selected=schedule.algorithm,
        fallback_reason=None,
        validation_passed=validation_passed,
        validation_warning=validation_warning,
        total_graph_replays=calibration_warmup + samples + 1 + warmup + samples + 1,
        profile_graph_replays=2,
        feedback_wall_seconds=time.perf_counter() - started,
        schedule=feedback,
        node_profile_files_by_rank=node_profile_files,
        applied_optimization=applied_optimization,
    )
    save(report)
    logger.info(
        "CV parallel feedback: total_graph_replays=%d validation_passed=%s",
        report["total_graph_replays"],
        validation_passed,
    )
    _log_final_optimization(schedule.algorithm, applied_optimization)
    logger.info(
        "CV parallel scheduled measured: algorithm=%s median_ms=%.3f delta_fb_ms=%.3f overlap=%s delta=%s",
        schedule.algorithm,
        measurement["median_ms"],
        elapsed_delta,
        overlap,
        delta,
    )
    logger.info(
        "CV parallel CV guard: cv=%s directional=%s same_type=%s delta_ms_by_rank=%s "
        "cube_delta_pp_by_rank=%s vector_delta_pp_by_rank=%s; report=%s",
        cv_verified,
        directional_coverage_verified,
        same_type_verified,
        [item["delta_ms"] for item in cv_validation],
        [item["cube_covered_by_vector_delta_percentage_points"] for item in cv_validation],
        [item["vector_covered_by_cube_delta_percentage_points"] for item in cv_validation],
        output,
    )
    if not validation_passed:
        logger.warning(
            "Scheduled validation warning (%s, F/B delta %.3f ms); keeping scheduled candidate. Report: %s",
            validation_warning,
            elapsed_delta,
            output,
        )
    return candidate
