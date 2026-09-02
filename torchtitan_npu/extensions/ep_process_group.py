# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Separate EP communication from dense and FSDP communication.

The registered sparse SPMD mesh is copied with its EP axis pointing at a
dedicated HCCL process group before TorchTitan enters
``maybe_set_sparse_mesh()``. This keeps the runtime mesh a real ``DeviceMesh``
and avoids patching dispatcher collectives. Dispatcher wiring is also patched
for EP implementations that access ``self.ep_mesh`` directly.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

_EP_MESHES: dict[str, DeviceMesh] = {}
_SPARSE_MESHES: dict[tuple[str, ...], DeviceMesh] = {}
_DEEPEP_GROUPS: dict[str, tuple[ProcessGroup, int]] = {}


def _is_fake_group(group: Any) -> bool:
    return dist.is_initialized() and str(dist.get_backend(group)).lower() == "fake"


def _create_separate_ep_mesh(ep_mesh: DeviceMesh) -> DeviceMesh:
    from torch_npu._C._distributed_c10d import (  # pyrefly: ignore[missing-import]
        ProcessGroupHCCL,
    )

    return ep_mesh._unflatten(
        0,
        (ep_mesh.size(),),
        ("ep",),
        backend_override={"ep": ProcessGroupHCCL.Options()},
    )


def _warm_spmd_mesh_axis(ep_mesh: DeviceMesh) -> None:
    """Convert a new EP PG before it can be observed by ``torch.compile``."""
    group = ep_mesh.get_group()
    if isinstance(group, dist.ProcessGroup):
        import spmd_types as spmd

        spmd.MeshAxis.of(group)


def _separate_ep_mesh(ep_mesh: DeviceMesh | None) -> DeviceMesh | None:
    """Return an EP-axis mesh backed by a dedicated HCCL process group."""
    if ep_mesh is None or ep_mesh.size() <= 1:
        return ep_mesh

    source_group = ep_mesh.get_group()
    source_key = source_group.group_name
    if _is_fake_group(source_group):
        return ep_mesh
    if any(mesh is ep_mesh for mesh in _EP_MESHES.values()):
        return ep_mesh
    if source_key in _EP_MESHES:
        return _EP_MESHES[source_key]

    separate_mesh = _create_separate_ep_mesh(ep_mesh)
    _warm_spmd_mesh_axis(separate_mesh)
    _EP_MESHES[source_key] = separate_mesh
    logger.info(
        "Created dedicated EP process group: source=%s dedicated=%s",
        source_key,
        separate_mesh.get_group().group_name,
    )
    return separate_mesh


def _replace_sparse_ep_mesh(sparse_mesh: DeviceMesh | None) -> DeviceMesh | None:
    """Return ``sparse_mesh`` with its EP axis using the dedicated EP PG."""
    if sparse_mesh is None or sparse_mesh.mesh_dim_names is None:
        return sparse_mesh
    if "ep" not in sparse_mesh.mesh_dim_names:
        return sparse_mesh

    source_ep_mesh = sparse_mesh["ep"]
    sparse_key = tuple(sparse_mesh._dim_group_names)
    cached = _SPARSE_MESHES.get(sparse_key)
    if cached is not None:
        return cached

    separate_ep_mesh = _separate_ep_mesh(source_ep_mesh)
    if separate_ep_mesh is source_ep_mesh:
        return sparse_mesh
    assert separate_ep_mesh is not None

    # DeviceMesh does not expose a public API for swapping one dimension's
    # process group.  Copying the mesh and its private group-name table keeps
    # all other sparse axes unchanged while reusing the already-created EP PG.
    separate_sparse_mesh = copy.copy(sparse_mesh)
    separate_sparse_mesh._dim_group_names = list(sparse_mesh._dim_group_names)
    ep_index = sparse_mesh.mesh_dim_names.index("ep")
    separate_sparse_mesh._dim_group_names[ep_index] = separate_ep_mesh._dim_group_names[0]
    _SPARSE_MESHES[sparse_key] = separate_sparse_mesh
    return separate_sparse_mesh


def _patch_sparse_mesh_registration() -> None:
    """Register the sparse mesh after replacing its EP process group."""
    import torchtitan.distributed.spmd_types as spmd_types

    original = spmd_types.set_spmd_meshes

    def patched(*, dense_mesh, sparse_mesh):
        original(
            dense_mesh=dense_mesh,
            sparse_mesh=_replace_sparse_ep_mesh(sparse_mesh),
        )

    cast("Any", spmd_types).set_spmd_meshes = patched


def _patch_base_dispatcher_wiring() -> None:
    """Wire every standard EP dispatcher to the separated EP mesh."""
    from torchtitan.models.common.token_dispatcher import BaseEPTokenDispatcher

    original = BaseEPTokenDispatcher.wire_meshes

    def patched(self, *, ep_mesh=None, **kwargs):
        if not getattr(self, "_uses_custom_ep_process_group", False):
            ep_mesh = _separate_ep_mesh(ep_mesh)
        return original(self, ep_mesh=ep_mesh, **kwargs)

    BaseEPTokenDispatcher.wire_meshes = patched


def _create_deepep_group(
    source_group: ProcessGroup,
    *,
    hidden_dim: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    top_k: int,
) -> ProcessGroup:
    """Create a dedicated HCCL group configured for the NPU DeepEP buffer."""
    import torch_npu
    from cann_ops_transformer import ElasticBuffer

    source_key = source_group.group_name
    ccl_buffer_size_mb = ElasticBuffer.get_moe_ep_ccl_buffer_size(
        source_group.size(),
        num_max_tokens_per_rank,
        hidden_dim,
        num_experts,
        top_k,
    )
    cached = _DEEPEP_GROUPS.get(source_key)
    if cached is not None:
        cached_group, configured_size_mb = cached
        if ccl_buffer_size_mb > configured_size_mb:
            raise RuntimeError(
                "The cached NPU DeepEP process group buffer is too small: "
                f"configured={configured_size_mb} MB, required={ccl_buffer_size_mb} MB. "
                "All dispatchers sharing an EP group must use a common worst-case "
                "configuration."
            )
        return cached_group

    ranks = dist.get_process_group_ranks(source_group)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": ccl_buffer_size_mb}
    deepep_group = cast(
        "ProcessGroup",
        dist.new_group(
            ranks=ranks,
            backend="hccl",
            pg_options=options,
            group_desc="npu_deepep",
            use_local_synchronization=True,
        ),
    )
    _DEEPEP_GROUPS[source_key] = (deepep_group, ccl_buffer_size_mb)
    logger.info(
        "Created dedicated NPU DeepEP process group: source=%s deepep=%s hccl_buffer_size=%d MB",
        source_key,
        deepep_group.group_name,
        ccl_buffer_size_mb,
    )
    return deepep_group


def get_deepep_group(
    ep_mesh: DeviceMesh,
    *,
    hidden_dim: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    top_k: int,
) -> ProcessGroup:
    """Return the dedicated NPU DeepEP group for an EP mesh."""
    if ep_mesh.size() <= 1:
        raise ValueError("NPU DeepEP requires an EP mesh with more than one rank.")
    return _create_deepep_group(
        ep_mesh.get_group(),
        hidden_dim=hidden_dim,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_experts=num_experts,
        top_k=top_k,
    )


_patch_sparse_mesh_registration()
_patch_base_dispatcher_wiring()
