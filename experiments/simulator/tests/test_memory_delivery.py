# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
from collections import Counter

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
            "--simulation.output-dir",
            str(directory),
            "--simulation.output-formats",
            "mem",
        ]
    )
    return workload, directory


def test_native_cli_exports_only_memory_and_preserves_raw_graph(exported_capture):
    workload, directory = exported_capture
    report = validate_workload(workload)
    annotations = workload.iteration.schedule.annotations
    assert annotations["capture_coordinates"]["logical_world_size"] == 4
    assert annotations["rank_table"]["world_size"] == 4
    assert annotations["capture_validation"] == report
    assert {path.name for path in directory.iterdir()} == {"memory"}
    memory = json.loads((directory / "memory/memory_summary.json").read_text())
    assert memory == workload.iteration.schedule.annotations["memory_summary"]
    assert set(memory) == {
        "steady_state_active_bytes_peak",
        "steady_state_forward_active_bytes_peak",
        "steady_state_backward_active_bytes_peak",
        "steady_state_optimizer_active_bytes_peak",
        "persistent_param_bytes",
        "optimizer_state_bytes",
        "activation_offload_by_layer",
    }
    with (directory / "memory/memory_timeline.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    active = peak = 0
    for row in rows:
        active += int(row["num_bytes"]) * (1 if row["action"] == "alloc" else -1)
        assert active == int(row["active_bytes_after"])
        peak = max(peak, active)
    assert peak == report["peak_active_bytes"]
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    views = [node for node in nodes.values() if node.annotations.get("metadata_view")]
    assert views
    source_ids = {node.op_id for node in views}
    assert source_ids <= set(annotations["native_dispatch_op_ids"])
    assert any(node.predecessors and node.successors for node in views)
    assert all(node.flops == 0 for node in views)


def test_native_mem_summary_includes_optimizer_state_and_steady_phase_peaks(exported_capture):
    workload, directory = exported_capture
    summary = json.loads((directory / "memory/memory_summary.json").read_text())
    layouts = workload.iteration.schedule.annotations["parameter_layouts"]
    # Native fused AdamW: two FP32 moments plus one device FP32 step per parameter.
    expected_state_bytes = 2 * sum(item["storage_bytes"] for item in layouts.values()) + 4 * len(layouts)
    assert summary["optimizer_state_bytes"] == expected_state_bytes
    plan = workload.iteration.schedule.annotations["memory_plan"]
    assert summary["persistent_param_bytes"] == sum(item["storage_bytes"] for item in layouts.values())
    assert plan.optimizer_state_storage_count == 3 * len(layouts)
    for phase in ("forward", "backward"):
        assert summary[f"steady_state_{phase}_active_bytes_peak"] == (
            getattr(plan, f"{phase}_peak_active_bytes") + expected_state_bytes
        )
    assert summary["steady_state_active_bytes_peak"] == max(
        summary[f"steady_state_{phase}_active_bytes_peak"] for phase in ("forward", "backward", "optimizer")
    )
    with (directory / "memory/tensor_lifetimes.csv").open() as stream:
        lifetimes = list(csv.DictReader(stream))
    end = max(int(row["death_seq"]) for row in lifetimes)
    deltas = Counter()
    for row in lifetimes:
        size = int(row["modeled_num_bytes"] or row["num_bytes"])
        state = row["kind"] == "optimizer_state"
        deltas[-1 if state else int(row["birth_seq"])] += size
        deltas[(end if state else int(row["death_seq"])) + 1] -= size
    with (directory / "memory/memory_events.csv").open() as stream:
        optimizer_start = min(int(row["seq_idx"]) for row in csv.DictReader(stream) if row["phase"] == "optimizer")
    active = peak = optimizer_peak = 0
    for seq in sorted(set(deltas) | {optimizer_start}):
        active += deltas[seq]
        peak = max(peak, active)
        if seq >= optimizer_start:
            optimizer_peak = max(optimizer_peak, active)
    assert peak == summary["steady_state_active_bytes_peak"]
    assert optimizer_peak == summary["steady_state_optimizer_active_bytes_peak"]
    trace = json.loads((directory / "memory/memory_trace.json").read_text())
    assert trace["metadata"]["timeline_scope"] == "captured_iteration"
    for key in (
        "optimizer_state_bytes",
        "steady_state_active_bytes_peak",
        "steady_state_forward_active_bytes_peak",
        "steady_state_backward_active_bytes_peak",
        "steady_state_optimizer_active_bytes_peak",
    ):
        assert trace["metadata"][key] == summary[key]


@pytest.mark.parametrize(
    "corruption", ["reference", "cycle", "peak", "saved_identity", "capture_rank", "steady_peak", "optimizer_state"]
)
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
    elif corruption == "steady_peak":
        monkeypatch.setattr(memory, "steady_state_peak_active_bytes", memory.steady_state_peak_active_bytes + 1)
        message = "steady-state lifetime sweep peak differs"
    elif corruption == "optimizer_state":
        monkeypatch.setattr(memory, "optimizer_state_bytes", memory.optimizer_state_bytes + 1)
        message = "optimizer state bytes differ"
    elif corruption == "capture_rank":
        monkeypatch.setitem(workload.schedule_plan.annotations, "capture_process_rank", 99)
        message = "plan capture worker differs"
    else:
        slot = next(slot for slot in memory.autograd_saved_tensors if slot.unpack_seq >= 0)
        monkeypatch.setattr(slot, "tensor_id", -123456789)
        message = "saved value lacks captured provenance"
    with pytest.raises(RuntimeError, match=message):
        validate_workload(workload)


@pytest.mark.parametrize("arguments", [["--simulation.output-formats", "json"], ["--simulation.csv-max-ranks", "1"]])
def test_removed_export_options_fail_before_native_model_build(arguments):
    with pytest.raises(SystemExit) as error:
        main(["--build-only", *arguments])
    assert error.value.code == 2


def test_empty_formats_have_no_filesystem_effects(exported_capture, tmp_path):
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import export_result

    destination = tmp_path / "absent"
    workload, _ = exported_capture
    export_result(workload, SimulationConfig(output_dir=str(destination)))
    assert not destination.exists()
    with pytest.raises(ValueError, match="Unsupported output formats"):
        export_result(workload, SimulationConfig(output_dir=str(destination), output_formats=["csv"]))
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("family", ["deepseek_v4_smoketest", "kimi_k3_smoketest"], ids=["dsv4", "kimi"])
def test_mxfp8_offload_and_shard_dtype_projection_preserve_native_execution(mode, family):
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import simulate

    common = dict(
        config=family,
        world_size=4,
        expert_parallel_degree=2,
        num_mtp_modules=1 if family == "deepseek_v4_smoketest" else 0,
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
    resident_layers = before["memory_summary"]["activation_offload_by_layer"]
    projected_layers = after["memory_summary"]["activation_offload_by_layer"]
    expected_layers = ["layers.0", "layers.1", "layers.2", "layers.3"]
    if family == "deepseek_v4_smoketest":
        expected_layers.append("mtp_layers.0")
    assert [item["layer"] for item in resident_layers] == expected_layers
    assert all(item["transfer_bytes"] == 0 for item in resident_layers)
    assert all(set(item) == {"layer", "transfer_bytes"} for item in projected_layers)
    assert all(item["transfer_bytes"] > 0 for item in projected_layers if item["layer"] != "<unattributed>")
    assert [(event.raw_op_type, event.execution_kind) for event in left.raw_events] == [
        (event.raw_op_type, event.execution_kind) for event in right.raw_events
    ]
    assert left.optimizer_state_bytes == right.optimizer_state_bytes
    assert left.optimizer_state_storage_count == right.optimizer_state_storage_count
    assert left.persistent_param_bytes == 2 * right.persistent_param_bytes
    assert right.model_active_bytes_peak < left.model_active_bytes_peak
    assert right.activation_offload_tensors or right.checkpoint_tensors
