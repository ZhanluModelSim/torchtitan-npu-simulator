# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json

import pytest
from torchtitan_simulator.cli import main
from torchtitan_simulator.validation import validate_workload


@pytest.fixture(scope="module")
def exported_capture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native_exports")
    workload = main(
        [
            "--simulation.world-size",
            "4",
            "--parallelism.expert-parallel-degree",
            "2",
            "--training.num-mtp-modules",
            "1",
            "--simulation.csv-max-ranks",
            "1",
            "--simulation.output-dir",
            str(directory),
            "--simulation.output-formats",
            "text",
            "html",
            "dot",
            "json",
            "csv",
            "mem",
        ]
    )
    return workload, directory


def test_native_cli_exports_preserve_plan_and_memory_with_limited_csv_ranks(exported_capture):
    workload, directory = exported_capture
    report = validate_workload(workload)
    annotations = workload.iteration.schedule.annotations
    assert annotations["capture_coordinates"]["logical_world_size"] == 4
    assert annotations["rank_table"]["world_size"] == 4
    assert annotations["capture_validation"] == report
    manifest = json.loads((directory / "export_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["capture_coordinates"]["logical_world_size"] == 4
    assert manifest["csv_max_ranks"] == 1
    assert len(list((directory / "kernel_summary").glob("rank_*.csv"))) == 1
    with (directory / "kernel_summary/rank_0.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["op_id"]) for row in rows} == {
        node.op_id
        for template in workload.step_templates.values()
        for node in template.nodes.values()
        if not node.annotations.get("metadata_view")
    }
    graph = json.loads((directory / "simulation_result.json").read_text())
    for key, template in graph["step_templates"].items():
        assert template["step_id"] == key
    assert {action["template_ref"] for action in graph["schedule_plan"]["actions"] if action["template_ref"]} <= set(
        graph["step_templates"]
    )
    memory = json.loads((directory / "memory/memory_summary.json").read_text())
    assert memory["active_bytes_peak"] == report["peak_active_bytes"]
    assert (
        graph["iteration"]["schedule"]["annotations"]["memory_plan"]["peak_active_bytes"] == report["peak_active_bytes"]
    )
    assert (directory / "schedule_plan.csv").read_text().startswith("seq_idx,schedule_order,action_id")
    assert "# DataSlots" in (directory / "schedule_plan.csv").read_text()
    for suffix in ("txt", "html", "dot"):
        assert (directory / f"simulation_result.{suffix}").stat().st_size > 100


@pytest.mark.parametrize("corruption", ["reference", "cycle", "peak", "saved_identity", "capture_rank"])
def test_validation_rejects_corrupted_native_capture(exported_capture, monkeypatch, corruption):
    workload, _ = exported_capture
    template = workload.step_templates["s0_F"]
    node = next(iter(template.nodes.values()))
    memory = workload.iteration.schedule.annotations["memory_plan"]
    if corruption == "reference":
        monkeypatch.setattr(node, "predecessors", [-123456789])
        message = "missing predecessor"
    elif corruption == "cycle":
        monkeypatch.setattr(node, "predecessors", [node.op_id])
        message = "dependency cycle"
    elif corruption == "peak":
        monkeypatch.setattr(memory, "peak_active_bytes", memory.peak_active_bytes + 1)
        message = "peak differs"
    elif corruption == "capture_rank":
        monkeypatch.setitem(workload.schedule_plan.annotations, "capture_process_rank", 99)
        message = "plan capture worker differs"
    else:
        slot = next(slot for slot in memory.autograd_saved_tensors if slot.unpack_seq >= 0)
        monkeypatch.setattr(slot, "tensor_id", -123456789)
        message = "saved value lacks captured provenance"
    with pytest.raises(RuntimeError, match=message):
        validate_workload(workload)


def test_invalid_csv_limit_fails_before_native_model_build():
    with pytest.raises(ValueError, match="csv_max_ranks must be positive"):
        main(["--build-only", "--simulation.csv-max-ranks", "0"])


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_mxfp8_offload_and_shard_dtype_projection_preserve_native_execution(mode):
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import simulate

    common = dict(
        world_size=4,
        expert_parallel_degree=2,
        num_mtp_modules=1,
        local_batch_size=1,
        global_batch_size=32,
        enable_quantized_training=True,
        activation_checkpoint_mode=mode,
    )
    resident = simulate(SimulationConfig(**common))
    projected = simulate(
        SimulationConfig(**common, memory_parameter_storage_dtype="bfloat16", memory_offload_ac_saved_tensors=True)
    )
    validate_workload(resident)
    validate_workload(projected)
    before, after = (workload.iteration.schedule.annotations for workload in (resident, projected))
    assert before["gradient_validation"] == after["gradient_validation"]
    assert before["quantization_configuration"] == after["quantization_configuration"]
    assert after["batch_configuration"]["gradient_accumulation"] == 8
    assert after["batch_configuration"]["captured_microbatches"] == 1
    left, right = before["memory_plan"], after["memory_plan"]
    assert [(event.raw_op_type, event.execution_kind) for event in left.raw_events] == [
        (event.raw_op_type, event.execution_kind) for event in right.raw_events
    ]
    assert left.persistent_param_bytes == 2 * right.persistent_param_bytes
    assert right.model_active_bytes_peak < left.model_active_bytes_peak
    assert right.activation_offload_tensors or right.checkpoint_tensors
