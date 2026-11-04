# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import pytest
from torchtitan.distributed.flex_shard.dist_muon import DistMuon
from torchtitan_simulator.config import SimulationConfig, prepare_config
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.runtime import meta_runtime
from torchtitan_simulator.validation import validate_workload


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"world_size": 2},
        {"world_size": 4},
        {"world_size": 2, "data_parallel_replicate_degree": 2, "data_parallel_shard_degree": 1},
        {"world_size": 4, "data_parallel_replicate_degree": 2},
        {"world_size": 8, "data_parallel_replicate_degree": 2, "expert_parallel_degree": 2, "logical_rank": 5},
        {"world_size": 4, "expert_parallel_degree": 2},
        {"world_size": 8, "expert_parallel_degree": 2, "logical_rank": 7},
        {"world_size": 8, "expert_parallel_degree": 4, "num_mtp_modules": 1},
        {"world_size": 4, "expert_parallel_degree": 2, "activation_checkpoint_mode": "full"},
        {"world_size": 4, "expert_parallel_degree": 2, "memory_offload_ac_saved_tensors": True},
    ],
    ids=[
        "single",
        "fsdp",
        "fsdp4-ep1",
        "ddp",
        "hsdp",
        "hsdp-ep-rank5",
        "ep",
        "expert-shard-rank7",
        "ep4-mtp",
        "ep-full-ac",
        "ep-activation-offload",
    ],
)
def test_dsv4_muon_captures_native_updates_communication_and_state(kwargs):
    options = SimulationConfig(optimizer_name="Muon", muon_ns_steps=2, **kwargs)
    prepare_config(options)  # Load the normal NPU package patches before testing scoped restoration.
    original = (DistMuon._validate_parameter_storage, DistMuon._validate_plan_across_ranks, DistMuon._apply_update)
    workload = simulate(options)
    assert original == (
        DistMuon._validate_parameter_storage,
        DistMuon._validate_plan_across_ranks,
        DistMuon._apply_update,
    )
    assert validate_workload(workload)["graph_acyclic"]
    annotations = workload.iteration.schedule.annotations
    groups = annotations["optimizer_parameter_groups"]
    owners = {name: group["algorithm"] for group in groups for name in group["parameter_names"]}
    experts = [name for name in owners if ".inner_experts." in name]
    assert len(experts) == (15 if kwargs.get("num_mtp_modules") else 12)
    assert all(owners[name] == "DistMuon" for name in experts)
    assert owners["tok_embeddings.weight"] == "AdamW"
    nodes = list(workload.step_templates["s0_OPTIMIZER"].nodes.values())
    counts = Counter(node.annotations["raw_op_type"] for node in nodes)
    assert counts["aten._fused_adamw_.default"] == 1
    assert counts["aten.bmm.default"] > 0 and counts["aten.baddbmm.default"] > 0
    updates = [node for node in nodes if node.annotations.get("optimizer_algorithm") == "DistMuon"]
    assert all(node.annotations["raw_op_type"] == "aten.add_.Tensor" for node in updates)
    assert {name for node in updates for name in node.annotations["optimizer_updated_parameter_names"]} == {
        name for name, owner in owners.items() if owner == "DistMuon"
    }
    # One FP32 momentum for Muon; two FP32 moments and a scalar step for AdamW.
    layouts = annotations["parameter_layouts"]
    expected_state_bytes = sum(
        layouts[name]["storage_bytes"] * (1 if owner == "DistMuon" else 2) + (0 if owner == "DistMuon" else 4)
        for name, owner in owners.items()
    )
    assert annotations["memory_plan"].optimizer_state_bytes == expected_state_bytes
    assert "not_performed_single_logical_rank" in annotations["runtime_capabilities"]["muon_peer_plan_validation"]
    if kwargs.get("world_size", 1) == 1:
        assert counts["c10d.alltoall_base_.default"] == 0
        assert counts["aten.mm.default"] + counts["aten.bmm.default"] == 68 * 2
        assert counts["aten.addmm.default"] + counts["aten.baddbmm.default"] == 68 * 2 * 2
        # Removing a real update must fail even when weight decay still exists.
        missing = updates[0]
        names = missing.annotations.pop("optimizer_updated_parameter_names")
        try:
            with pytest.raises(RuntimeError, match="optimizer update set differs"):
                validate_workload(workload)
        finally:
            missing.annotations["optimizer_updated_parameter_names"] = names
        names.append(names[0])
        try:
            with pytest.raises(RuntimeError, match="parameter updated more than once"):
                validate_workload(workload)
        finally:
            names.pop()
        algorithm = missing.annotations["optimizer_algorithm"]
        missing.annotations["optimizer_algorithm"] = "AdamW"
        try:
            with pytest.raises(RuntimeError, match="optimizer update algorithm differs"):
                validate_workload(workload)
        finally:
            missing.annotations["optimizer_algorithm"] = algorithm
    elif kwargs.get("data_parallel_shard_degree") == 1:
        assert counts["c10d.alltoall_base_.default"] == 0
    else:
        assert counts["c10d.alltoall_base_.default"] > 0


def test_muon_meta_adaptation_restores_after_exception():
    prepared = prepare_config(SimulationConfig(optimizer_name="Muon"))
    original = (DistMuon._validate_parameter_storage, DistMuon._validate_plan_across_ranks, DistMuon._apply_update)
    with pytest.raises(RuntimeError, match="intentional failure"), meta_runtime() as runtime:
        runtime.configure_model(prepared)
        assert DistMuon._validate_parameter_storage is not original[0]
        raise RuntimeError("intentional failure")
    assert original == (
        DistMuon._validate_parameter_storage,
        DistMuon._validate_plan_across_ranks,
        DistMuon._apply_update,
    )


@pytest.mark.parametrize("world_size", [8, 16])
def test_ep1_split_expert_matrix_is_rejected_before_optimizer_construction(monkeypatch, world_size):
    from torchtitan.components.optimizer import OptimizersContainer

    def unexpected_optimizer_construction(*args, **kwargs):
        pytest.fail("Unsupported Muon layout must be rejected before constructing the native optimizer")

    monkeypatch.setattr(OptimizersContainer, "__init__", unexpected_optimizer_construction)
    with pytest.raises(ValueError, match=r"Muon with EP=1 does not support Shard\(1\) -> Shard\(0\)") as error:
        simulate(SimulationConfig(optimizer_name="Muon", world_size=world_size))
    assert "layers.0.moe.routed_experts.inner_experts.w1_EFD" in str(error.value)
    assert "mesh axis='dp_shard'" in str(error.value)
    assert f"size={world_size}" in str(error.value)


def test_ep1_layout_guard_uses_actual_geometry_and_preserves_adamw():
    from torchtitan_simulator.model_overrides import ModelOverrides, MoEOverrides

    # Eight experts keep whole matrices on eight FSDP shards; world size alone
    # must not decide whether the layout is supported.
    muon = simulate(
        SimulationConfig(
            optimizer_name="Muon",
            muon_ns_steps=2,
            world_size=8,
            model_overrides=ModelOverrides(moe_args=MoEOverrides(num_experts=8)),
        )
    )
    assert validate_workload(muon)["graph_acyclic"]
    muon_layout = muon.iteration.schedule.annotations["parameter_layouts"]
    assert muon_layout["layers.0.moe.routed_experts.inner_experts.w1_EFD"]["local_shape"] == (1, 256, 256)

    adamw = simulate(SimulationConfig(optimizer_name="AdamW", world_size=8))
    assert validate_workload(adamw)["graph_acyclic"]
    adamw_layout = adamw.iteration.schedule.annotations["parameter_layouts"]
    assert adamw_layout["layers.0.moe.routed_experts.inner_experts.w1_EFD"]["local_shape"] == (4, 32, 256)
