# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
from torchtitan_simulator.capture.comm_events import (
    _group_name,
    _resolve_comm_ranks,
    _resolve_world_size,
    default_collective_context,
)


def test_functional_mesh_axis_resolves_actual_group_instead_of_tuple_length(monkeypatch):
    class Group:
        group_name = "expert-shard"

        def size(self):
            return 4

    group = Group()

    class Mesh:
        def get_group(self, axis):
            assert axis == "efsdp"
            return group

    monkeypatch.setattr(dist, "get_process_group_ranks", lambda pg: [8, 10, 12, 14] if pg is group else [])
    mesh_group = (Mesh(), "efsdp")
    # A default TP context must not overwrite an explicit expert-shard axis.
    with default_collective_context("tp", object()):
        assert _resolve_world_size(mesh_group) == 4
        assert _group_name(mesh_group) == "expert-shard"
        assert _resolve_comm_ranks(mesh_group) == [[8, 10, 12, 14]]
    assert _resolve_comm_ranks([8, 10, 12, 14]) == [[8, 10, 12, 14]]
