# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import multiprocessing as mp

import pytest
import torch.distributed as dist

from torchtitan_npu.simulator.meta_env import (
    _patch_new_group_for_fake_backend,
    _patch_parallel_dims_for_multi_proc,
    patch_device_type_to_meta,
    unpatch_device_type_to_meta,
)


def _logical_submesh_worker(
    rank: int,
    rendezvous_path: str,
    result_queue,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    patch_device_type_to_meta()
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=2,
        )
        _patch_new_group_for_fake_backend()
        _patch_parallel_dims_for_multi_proc(full_ws=4, gloo_ws=2)

        import torch.distributed.device_mesh as device_mesh

        world_mesh = device_mesh.init_device_mesh(
            "meta", (4,), mesh_dim_names=("world",)
        )
        sparse_mesh = world_mesh._unflatten(
            0,
            (2, 1, 1, 2, 1),
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
        )
        ep_mesh = sparse_mesh["ep"]
        result_queue.put(
            (
                rank,
                ep_mesh.mesh.tolist(),
                ep_mesh.get_coordinate(),
                dist.get_process_group_ranks(ep_mesh.get_group()),
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        unpatch_device_type_to_meta()


@pytest.fixture
def single_gloo_rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29917"
    patch_device_type_to_meta()
    dist.init_process_group("gloo", rank=0, world_size=1)
    _patch_new_group_for_fake_backend()
    _patch_parallel_dims_for_multi_proc(full_ws=4, gloo_ws=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
        unpatch_device_type_to_meta()


def test_oversized_fake_group_registers_logical_rank_map(single_gloo_rank):
    import torch.distributed.device_mesh as device_mesh
    from torch.distributed import distributed_c10d as c10d

    world_mesh = device_mesh.init_device_mesh(
        "meta", (4,), mesh_dim_names=("world",)
    )
    group = world_mesh.get_group()

    assert dist.get_backend(group) == "fake"
    assert group in c10d._world.pg_map
    assert c10d._world.pg_group_ranks[group] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert dist.get_group_rank(group, 0) == 0
    assert dist.get_global_rank(group, 3) == 3


def test_only_pp_dimension_reuses_real_gloo_group(single_gloo_rank):
    import torch.distributed.device_mesh as device_mesh

    world_mesh = device_mesh.init_device_mesh(
        "meta", (4,), mesh_dim_names=("world",)
    )
    logical_mesh = world_mesh._unflatten(
        0,
        (1, 4),
        ("pp", "cp"),
    )

    assert dist.get_backend(logical_mesh["pp"].get_group()) == "gloo"
    assert dist.get_backend(logical_mesh["cp"].get_group()) == "fake"


def test_patched_new_group_preserves_positional_arguments(single_gloo_rank):
    group = dist.new_group(
        [0], None, "fake", None, False, "positional-arguments", None, False
    )

    assert dist.get_backend(group) == "fake"
    assert group.size() == 1


def test_out_of_range_real_group_is_not_silently_converted(single_gloo_rank):
    with pytest.raises(ValueError, match="world size|out of range"):
        dist.new_group(ranks=[0, 1], backend="gloo")


def test_each_pp_worker_selects_its_logical_submesh(tmp_path):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    rendezvous_path = str(tmp_path / "logical-submesh-rendezvous")
    processes = [
        context.Process(
            target=_logical_submesh_worker,
            args=(rank, rendezvous_path, result_queue),
        )
        for rank in range(2)
    ]

    for process in processes:
        process.start()
    results = [result_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert sorted(results) == [
        (0, [0, 1], (0,), [0, 1]),
        (1, [2, 3], (0,), [2, 3]),
    ]
