# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real single-rank CPU meshes for local model and metadata tests."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch.distributed as dist
from torchtitan.distributed import parallelism_context, spmd_types


@contextmanager
def single_rank_spmd_context():
    """Activate CPU DP/CP/TP axes and restore the caller's runtime state."""
    owns_group = not dist.is_initialized()
    previous_mesh_state = dict(spmd_types._MESH_TLS.__dict__)
    with TemporaryDirectory(prefix="torchtitan-cpu-spmd-") as directory:
        try:
            if owns_group:
                dist.init_process_group(
                    "gloo", init_method=Path(directory, "rendezvous").as_uri(), rank=0, world_size=1
                )
            if dist.get_world_size() != 1:
                raise RuntimeError("The local model fixture requires a single-rank process group.")
            context = parallelism_context.ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=1,
                enable_sequence_parallel=False,
            )
            # Package import may select NPU globally; these tests execute real CPU arithmetic.
            # Keep even singleton axes on Gloo rather than the optional fake-PG backend.
            with (
                patch.object(parallelism_context, "device_type", "cpu"),
                patch.object(parallelism_context.ParallelismContext, "_axis_is_active", return_value=True),
            ):
                context.build_mesh()
            with context.activate_spmd():
                yield context
        finally:
            spmd_types._MESH_TLS.__dict__.clear()
            spmd_types._MESH_TLS.__dict__.update(previous_mesh_state)
            if owns_group and dist.is_initialized():
                dist.destroy_process_group()


@contextmanager
def gloo_ep_group_context():
    """Create real dedicated Gloo groups at the NPU EP-group creation boundary."""
    from torch.distributed.device_mesh import DeviceMesh

    from torchtitan_npu.extensions.torchtitan import ep_process_group

    previous_mesh_state = dict(spmd_types._MESH_TLS.__dict__)
    previous_ep_meshes = ep_process_group._EP_MESHES.copy()
    previous_sparse_meshes = ep_process_group._SPARSE_MESHES.copy()
    groups = []

    def create_gloo_mesh(source_mesh):
        group = dist.new_group(
            ranks=dist.get_process_group_ranks(source_mesh.get_group()),
            backend="gloo",
            group_desc="cpu_unit_ep",
            use_local_synchronization=True,
        )
        groups.append(group)
        return DeviceMesh.from_group(group, "cpu", mesh_dim_names=("ep",))

    try:
        with patch.object(ep_process_group, "_create_separate_ep_mesh", create_gloo_mesh):
            yield
    finally:
        spmd_types._MESH_TLS.__dict__.clear()
        spmd_types._MESH_TLS.__dict__.update(previous_mesh_state)
        ep_process_group._EP_MESHES.clear()
        ep_process_group._EP_MESHES.update(previous_ep_meshes)
        ep_process_group._SPARSE_MESHES.clear()
        ep_process_group._SPARSE_MESHES.update(previous_sparse_meshes)
        for group in reversed(groups):
            dist.destroy_process_group(group)
