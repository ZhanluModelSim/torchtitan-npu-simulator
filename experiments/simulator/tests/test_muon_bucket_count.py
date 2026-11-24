# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan.distributed.flex_shard.dist_muon import DistMuon
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize(
    "ep, count, expected",
    [(1, 1, ("all",)), (1, 3, ("bucket_1", "bucket_2", "bucket_3")), (2, 2, ("bucket_1", "bucket_2"))],
    ids=["single", "three", "two-transport-groups"],
)
def test_configured_muon_count_builds_nonempty_native_plans(monkeypatch, ep, count, expected):
    original = DistMuon._initialize_plan
    observed = []

    def initialize_plan(optimizer, layouts):
        original(optimizer, layouts)
        observed.append(tuple(spec.name for spec in optimizer._specs))
        assert len(optimizer._bucket_plans) == count

    monkeypatch.setattr(DistMuon, "_initialize_plan", initialize_plan)
    simulate(
        SimulationConfig(
            optimizer_name="Muon", muon_ns_steps=1, muon_bucket_count=count, world_size=8, expert_parallel_degree=ep
        )
    )
    assert observed == [expected]


def test_muon_bucket_count_cannot_merge_distinct_transport_groups():
    with pytest.raises(ValueError, match="too small for 2 distinct transport groups"):
        simulate(
            SimulationConfig(
                optimizer_name="Muon", muon_ns_steps=1, muon_bucket_count=1, world_size=8, expert_parallel_degree=2
            )
        )
