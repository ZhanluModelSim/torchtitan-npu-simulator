# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan_npu.simulator.rank_context import SimulationRankContext


def test_capture_rank_maps_to_representative_pp_logical_rank(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    context = SimulationRankContext.resolve(
        logical_world_size=64,
        pp_degree=4,
    )

    assert context.capture_process_rank == 3
    assert context.capture_world_size == 4
    assert context.logical_global_rank == 48


def test_single_process_non_pp_uses_logical_rank_zero(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    context = SimulationRankContext.resolve(
        logical_world_size=32,
        pp_degree=1,
    )

    assert context.capture_process_rank == 0
    assert context.capture_world_size == 1
    assert context.logical_global_rank == 0


def test_pp_capture_rejects_implicit_rank_zero(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="RANK/WORLD_SIZE are absent"):
        SimulationRankContext.resolve(
            logical_world_size=64,
            pp_degree=4,
        )


def test_pp_capture_rejects_partial_environment(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="must either both be set"):
        SimulationRankContext.resolve(
            logical_world_size=64,
            pp_degree=4,
        )
