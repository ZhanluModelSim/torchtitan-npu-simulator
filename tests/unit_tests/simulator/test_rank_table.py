# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch.distributed as dist

from torchtitan_npu.simulator.rank_table import build_rank_table


@pytest.fixture
def fake_world():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29812"
    dist.init_process_group("fake", rank=0, world_size=16)
    yield
    dist.destroy_process_group()


def test_build_rank_table_matches_real_parallel_dims_mesh(fake_world):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=16, cp=1, tp=1, pp=1, ep=8, world_size=16)
    parallel_dims.build_mesh()

    table = build_rank_table(parallel_dims)

    assert table.world_size == 16
    assert table.dim_degrees["ep"] == 8
    # verified by hand in design doc §5.6: ep groups are contiguous blocks
    ep_groups = sorted(table.process_groups["ep"], key=lambda g: g[0])
    assert ep_groups == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    assert table.rank_coordinates[0]["ep"] == 0
    assert table.rank_coordinates[8]["ep"] == 0
    assert table.rank_coordinates[9]["ep"] == 1


def test_build_rank_table_every_rank_has_coordinates_for_every_group_dim(fake_world):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=16, cp=1, tp=1, pp=1, ep=8, world_size=16)
    parallel_dims.build_mesh()

    table = build_rank_table(parallel_dims)
    for rank in range(16):
        assert rank in table.rank_coordinates
        for dim_name in table.process_groups:
            assert dim_name in table.rank_coordinates[rank]


def test_rank_table_to_dict_is_json_serializable(fake_world):
    import json

    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=16, cp=1, tp=1, pp=1, ep=8, world_size=16)
    parallel_dims.build_mesh()

    table = build_rank_table(parallel_dims)
    serialized = json.dumps(table.to_dict())
    assert '"world_size": 16' in serialized
