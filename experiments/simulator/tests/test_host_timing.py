# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host diagnostics remain separate from graph execution facts and sessions."""

import json
import logging

import pytest
from torchtitan_simulator import host_timing
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.host_timing import HostTimings
from torchtitan_simulator.runner import simulate


def test_host_scopes_accumulate_inclusive_time_and_return_independent_snapshots(monkeypatch):
    clock = iter([0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    monkeypatch.setattr(host_timing.time, "perf_counter", lambda: next(clock))
    timings = HostTimings()

    with timings.measure("total"):
        with timings.measure("phase"):
            pass
        with timings.measure("phase"):
            pass
    summary = timings.summary("completed")

    assert summary["stages"] == {
        "phase": {"wall_seconds": 2.0, "calls": 2},
        "total": {"wall_seconds": 6.0, "calls": 1},
    }
    summary["stages"]["phase"]["calls"] = 99
    assert timings.summary("completed")["stages"]["phase"]["calls"] == 2
    assert HostTimings().summary("completed")["stages"] == {}


def test_failed_host_scope_records_elapsed_time_without_swallowing_error(monkeypatch):
    clock = iter([7.0, 9.5])
    monkeypatch.setattr(host_timing.time, "perf_counter", lambda: next(clock))
    timings = HostTimings()

    with pytest.raises(ValueError, match="scope failed"), timings.measure("configuration"):
        raise ValueError("scope failed")

    assert timings.summary("failed")["stages"] == {"configuration": {"wall_seconds": 2.5, "calls": 1}}


def test_simulate_reports_failure_and_isolates_muon_then_adamw_host_stages(caplog):
    caplog.set_level(logging.INFO, logger="torchtitan_simulator.runner")
    with pytest.raises(ValueError):
        simulate(SimulationConfig(world_size=0))
    failure = json.loads(
        next(
            record.message.split(": ", 1)[1]
            for record in caplog.records
            if record.message.startswith("Simulation host timings: ")
        )
    )
    assert failure["status"] == "failed"
    assert failure["stages"]["configuration"]["calls"] == 1
    assert failure["stages"]["total"]["calls"] == 1

    summaries = []
    for algorithm in ("Muon", "AdamW"):
        options = SimulationConfig(
            optimizer_name=algorithm,
            muon_ns_steps=1 if algorithm == "Muon" else None,
            world_size=2,
            seq_len=128,
        )
        workload = simulate(options)
        annotations = workload.iteration.schedule.annotations
        timing = annotations["host_timings"]
        summaries.append(timing)
        assert timing["status"] == "completed"
        stages = timing["stages"]
        for phase in (
            "configuration",
            "model_build",
            "parallelize",
            "optimizer_build",
            "warmup",
            "capture_step",
            "total",
        ):
            assert stages[phase]["calls"] == 1
            assert stages[phase]["wall_seconds"] >= 0
        for phase in ("forward", "backward"):
            assert stages[f"warmup.{phase}"]["calls"] == 1
            assert stages[f"capture.{phase}"]["calls"] == 1
        assert (
            stages["capture_step.estimate_memory"]["wall_seconds"] == annotations["capture_timings"]["estimate_memory"]
        )
        assert stages["total"]["wall_seconds"] >= stages["optimizer_build"]["wall_seconds"]
        assert annotations["memory_plan"].optimizer_state_bytes > 0
    assert any(name.startswith("optimizer_planning.") for name in summaries[0]["stages"])
    assert not any(name.startswith("optimizer_planning.") for name in summaries[1]["stages"])
