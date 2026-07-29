# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.comm_group_resolver import (
    CommGroupResolver,
    resolve_comm_event_groups,
)
from torchtitan_npu.simulator.rank_table import RankTable


def _rank_table() -> RankTable:
    return RankTable(
        world_size=8,
        dim_degrees={"tp": 2, "ep": 4, "dp_shard": 2},
        process_groups={
            "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "ep": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "dp_shard": [[0, 4], [1, 5], [2, 6], [3, 7]],
        },
        dim_by_group_name={
            "pg_tp_0": "tp",
            "pg_ep_0": "ep",
            "pg_fsdp_0": "dp_shard",
        },
    )


def _event(
    group_name: str,
    comm_ranks: list[list[int]],
    *,
    op_id: int = 1,
) -> CommEvent:
    return CommEvent(
        event_id="comm_1",
        comm_primitive="allgather",
        group_name=group_name,
        raw_group_name=group_name,
        world_size=len(comm_ranks[0]) if comm_ranks else 1,
        tensor_shape=(8,),
        dtype="bfloat16",
        volume_bytes=16,
        op_id=op_id,
        comm_dim=group_name,
        comm_ranks=comm_ranks,
    )


def test_resolves_framework_group_name_to_semantic_dimension():
    resolver = CommGroupResolver(_rank_table())

    assert resolver.resolve("pg_tp_0", [[0, 1]]) == "tp"
    assert resolver.resolve("pg_ep_0", [[0, 1, 2, 3]]) == "ep"
    assert resolver.resolve("pg_fsdp_0", [[0, 4]]) == "fsdp"


def test_resolves_by_unique_rank_group_when_name_is_unknown():
    resolver = CommGroupResolver(_rank_table())

    assert resolver.resolve("opaque_123", [[4, 5, 6, 7]]) == "ep"


def test_falls_back_to_raw_name_when_rank_group_is_ambiguous():
    rank_table = _rank_table()
    rank_table.process_groups["cp"] = [[0, 1], [2, 3], [4, 5], [6, 7]]
    rank_table.dim_degrees["cp"] = 2
    resolver = CommGroupResolver(rank_table)

    assert resolver.resolve("opaque_123", [[0, 1]]) == "opaque_123"


def test_hsdp_uses_group_name_when_replicate_and_shard_ranks_overlap():
    rank_table = RankTable(
        world_size=4,
        dim_degrees={"dp_replicate": 2, "dp_shard": 2},
        process_groups={
            "dp_replicate": [[0, 1], [2, 3]],
            "dp_shard": [[0, 1], [2, 3]],
        },
        dim_by_group_name={
            "pg_replicate": "dp_replicate",
            "pg_shard": "dp_shard",
        },
    )
    resolver = CommGroupResolver(rank_table)

    assert resolver.resolve("pg_replicate", [[0, 1]]) == "dp_replicate"
    assert resolver.resolve("pg_shard", [[0, 1]]) == "fsdp"
    assert resolver.resolve("opaque_123", [[0, 1]]) == "opaque_123"


def test_world_sized_fsdp_ignores_non_model_rank_domains():
    rank_table = RankTable(
        world_size=4,
        dim_degrees={"batch": 4, "loss_mesh": 4, "fsdp": 4, "efsdp": 4},
        process_groups={
            "batch": [[0, 1, 2, 3]],
            "loss_mesh": [[0, 1, 2, 3]],
            "fsdp": [[0, 1, 2, 3]],
            "efsdp": [[0, 1, 2, 3]],
        },
    )
    resolver = CommGroupResolver(rank_table)

    assert resolver.resolve("0", [[0, 1, 2, 3]], "allgather") == "fsdp"
    assert resolver.resolve("0", [[0, 1, 2, 3]], "allreduce") == "0"


def test_enrichment_preserves_raw_name_and_updates_l0_mapping():
    event = _event("pg_fsdp_0", [[0, 4]], op_id=17)

    names_by_op_id = resolve_comm_event_groups([event], _rank_table())

    assert event.group_name == "fsdp"
    assert event.raw_group_name == "pg_fsdp_0"
    assert event.comm_dim == "fsdp"
    assert names_by_op_id == {17: ("fsdp", "pg_fsdp_0")}
