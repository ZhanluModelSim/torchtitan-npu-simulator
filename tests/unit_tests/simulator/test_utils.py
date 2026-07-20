# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchtitan_npu.simulator.utils import (
    get_world_size,
    resolve_simulation_runtime,
    resolve_simulation_runtime_from_environment,
)


def _config(
    *,
    pp: int = 1,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    etp: int = 1,
    dp_replicate: int = 1,
    dp_shard: int = -1,
    world_size: int | None = None,
    legacy_degrees: dict[str, int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallelism=SimpleNamespace(
            pipeline_parallel_degree=pp,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
            expert_parallel_degree=ep,
            expert_tensor_parallel_degree=etp,
            data_parallel_replicate_degree=dp_replicate,
            data_parallel_shard_degree=dp_shard,
        ),
        comm=SimpleNamespace(mode="default"),
        simulation=SimpleNamespace(
            world_size=world_size,
            simulated_parallel_degrees=dict(legacy_degrees or {}),
        ),
    )


def test_runtime_mode_is_derived_from_final_pp_degree() -> None:
    single = _config(pp=1, dp_shard=8)
    multi = _config(pp=4, dp_shard=2)

    single_runtime = resolve_simulation_runtime(single)
    multi_runtime = resolve_simulation_runtime(multi)

    assert single_runtime.comm_mode == single.comm.mode == "fake_backend"
    assert multi_runtime.comm_mode == multi.comm.mode == "multi_proc_meta"


def test_runtime_recomputes_auto_dp_and_replaces_stale_degree_snapshot() -> None:
    config = _config(
        pp=2,
        cp=4,
        dp_shard=-1,
        world_size=64,
        legacy_degrees={"pp": 4, "cp": 4, "dp_shard": 4, "world_size": 64},
    )

    runtime = resolve_simulation_runtime(config)

    assert config.parallelism.data_parallel_shard_degree == 8
    assert runtime.parallel_degrees == {
        "pp": 2,
        "tp": 1,
        "cp": 4,
        "ep": 1,
        "dp_replicate": 1,
        "dp_shard": 8,
        "etp": 1,
        "world_size": 64,
    }
    assert config.simulation.simulated_parallel_degrees == runtime.parallel_degrees


def test_ep_and_etp_are_not_dense_world_size_factors() -> None:
    config = _config(pp=4, cp=4, ep=4, dp_shard=8)

    assert get_world_size(config) == 128
    runtime = resolve_simulation_runtime(config)
    assert runtime.world_size == 128
    assert runtime.parallel_degrees["dp_shard"] == 8


def test_runtime_rejects_world_size_mismatch() -> None:
    config = _config(pp=4, cp=4, dp_shard=2, world_size=64)

    with pytest.raises(ValueError, match="parallel degrees require world_size 32"):
        resolve_simulation_runtime(config)


def test_runtime_rejects_invalid_expert_mesh() -> None:
    config = _config(pp=1, cp=2, ep=3, dp_shard=2)

    with pytest.raises(ValueError, match=r"EP\*ETP .* must divide"):
        resolve_simulation_runtime(config)


def test_ngpu_overrides_registry_world_size_before_auto_dp_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(pp=4, cp=4, dp_shard=-1, world_size=384)
    monkeypatch.setenv("NGPU", "64")
    monkeypatch.delenv("TORCHTITAN_SIM_WORLD_SIZE", raising=False)

    runtime = resolve_simulation_runtime_from_environment(config, cli_args=[])

    assert runtime.world_size == 64
    assert config.simulation.world_size == 64
    assert config.parallelism.data_parallel_shard_degree == 4


def test_explicit_cli_world_size_overrides_ngpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(pp=4, cp=4, dp_shard=-1, world_size=32)
    monkeypatch.setenv("NGPU", "64")

    runtime = resolve_simulation_runtime_from_environment(
        config,
        cli_args=["--simulation.world-size", "32"],
    )

    assert runtime.world_size == 32
    assert config.parallelism.data_parallel_shard_degree == 2
