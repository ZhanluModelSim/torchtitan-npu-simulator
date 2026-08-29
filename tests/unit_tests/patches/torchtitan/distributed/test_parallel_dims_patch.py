# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract test for the ParallelDims singleton patch."""

def test_parallel_dims_get_returns_the_instance_created_by_post_init(monkeypatch):
    from torchtitan.distributed.parallel_dims import ParallelDims

    # Importing the patch applies the same classmethod registration used by
    # package startup, while keeping this test independent of a process group.
    import torchtitan_npu.patches.torchtitan.distributed.parallel_dims  # noqa: F401

    monkeypatch.setattr(ParallelDims, "_global_instance", None, raising=False)
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
    )

    assert ParallelDims.get() is dims
