# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Actual FSDP group roles survive flattened meshes and shared process groups."""

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize("config", ["deepseek_v4_smoketest", "kimi_k3_smoketest"], ids=["dsv4", "kimi"])
@pytest.mark.parametrize(
    "world,replicate,rank,dense_ranks,expert_ranks,replicate_ranks",
    [
        (8, 1, 0, tuple(range(8)), (0, 2, 4, 6), ()),
        (16, 2, 0, tuple(range(8)), (0, 2, 4, 6), (0, 8)),
        (16, 2, 15, tuple(range(8, 16)), (9, 11, 13, 15), (7, 15)),
    ],
    ids=["shared-world-group", "flattened-hsdp-group", "flattened-hsdp-last-rank"],
)
def test_fsdp_capture_labels_actual_shard_and_replicate_groups(
    tmp_path, config, world, replicate, rank, dense_ranks, expert_ranks, replicate_ranks
):
    workload = simulate(
        SimulationConfig(
            config=config,
            world_size=world,
            data_parallel_replicate_degree=replicate,
            context_parallel_degree=2,
            expert_parallel_degree=2,
            logical_rank=rank,
            seq_len=256,
            num_mtp_modules=0,
            output_dir=str(tmp_path),
        )
    )
    events = workload.iteration.schedule.annotations["communication_events"]
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    shard_events = [event for event in events if event["fsdp_group_id"] and event["comm_primitive"] == "allgather"]
    reductions = [event for event in events if event["gradient_reduction"]]

    assert {tuple(event["comm_ranks"][0]) for event in shard_events} == {dense_ranks, expert_ranks}
    assert {tuple(event["comm_ranks"][0]) for event in reductions if event["comm_primitive"] == "reduce_scatter"} == {
        dense_ranks,
        expert_ranks,
    }
    assert {tuple(event["comm_ranks"][0]) for event in reductions if event["comm_primitive"] == "allreduce"} == (
        {replicate_ranks} if replicate_ranks else set()
    )
    for event in [*shard_events, *reductions]:
        expected = "dp_replicate" if event["comm_primitive"] == "allreduce" else "fsdp"
        node = nodes[event["op_id"]]
        assert event["group_name"] == event["comm_dim"] == expected
        assert node.annotations["group_name"] == node.annotations["comm_dim"] == expected
        assert node.annotations["raw_group_name"] == event["raw_group_name"]
        assert event["raw_group_name"] != expected

    # The same WORLD group can serve FSDP AG/RS and the loss all-reduce.
    loss_events = [
        event for event in events if event["comm_primitive"] == "allreduce" and not event["gradient_reduction"]
    ]
    assert loss_events and all(event["group_name"] == "loss_mesh" for event in loss_events)
    if replicate == 1:
        assert {event["raw_group_name"] for event in loss_events} <= {event["raw_group_name"] for event in shard_events}
    assert any(event["comm_dim"] == "ep" for event in events)
    assert any(event["comm_dim"] == "cp" for event in events)
