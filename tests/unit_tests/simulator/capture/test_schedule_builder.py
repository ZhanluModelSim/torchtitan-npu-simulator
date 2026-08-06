# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_graph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.rank_table import RankTable


def _rank_table() -> RankTable:
    return RankTable(
        world_size=4,
        dim_degrees={"ep": 2, "tp": 1, "pp": 1, "dp_replicate": 1, "fsdp": 2},
        rank_coordinates={
            0: {"ep": 0, "pp": 0, "dp_replicate": 0},
            1: {"ep": 1, "pp": 0, "dp_replicate": 0},
            2: {"ep": 0, "pp": 0, "dp_replicate": 0},
            3: {"ep": 1, "pp": 0, "dp_replicate": 0},
        },
        process_groups={"ep": [[0, 1], [2, 3]], "fsdp": [[0, 2], [1, 3]]},
        dim_by_group_name={"grp_ep": "ep", "grp_fsdp": "fsdp"},
    )


def test_build_schedule_graph_creates_rank_local_instance():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[],
    )
    assert len(graph.instances) == 1
    assert {i.instance_id for i in graph.instances} == {"rank0_tmpl"}
    assert all(i.step_ref == "tmpl" for i in graph.instances)


def test_build_schedule_graph_creates_rank_local_collective_data_pass():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    event = CommEvent(
        event_id="c1", comm_primitive="all_to_all", group_name="grp_ep",
        world_size=2, tensor_shape=(8, 16), dtype="bfloat16", volume_bytes=256,
    )
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[event],
    )
    # Capture is rank-local: a collective invocation is represented once and
    # retains its communication dimension/group metadata.
    assert len(graph.data_passes) == 1
    assert graph.data_passes[0].src_instance == "rank0"
    assert graph.data_passes[0].dst_instance == "group:"
    assert all(p.comm_primitive == "all_to_all" for p in graph.data_passes)
    assert all(p.requires_communication for p in graph.data_passes)


def test_build_schedule_graph_preserves_collective_with_unknown_group_name():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    event = CommEvent(
        event_id="c2", comm_primitive="allreduce", group_name="totally_unrecognized_group",
        world_size=4, tensor_shape=(4,), dtype="float32", volume_bytes=16,
    )
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[event],
    )
    assert len(graph.data_passes) == 1
    assert graph.data_passes[0].src_instance == "rank0"
    assert graph.data_passes[0].comm_primitive == "allreduce"


def test_build_schedule_graph_carries_rank_table_in_annotations():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[],
    )
    assert graph.annotations["rank_table"]["world_size"] == 4


def test_build_schedule_graph_degrees_from_rank_table():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[],
    )
    assert graph.pp_degree == 1
    assert graph.tp_degree == 1
    assert graph.dp_degree == 2  # dp_replicate(1) * fsdp(2)
