# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve model-owned expert layouts against real DTensor storage metadata."""

import importlib
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan.components.optim import AdamW
from torchtitan.distributed.flex_shard.dist_muon import _resolve_storage_to_compute_transition

from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu_recipes.models.deepseek_v4 import _dsv4_muon_profile
from torchtitan_npu_recipes.models.deepseek_v4_1 import _v41_muon_profile


@pytest.fixture
def sparse_storage_mesh():
    # Only metadata is consumed: FakeProcessGroup does not validate transport
    # or optimizer numerics. Rank 1 belongs to expert 1 in the EP dimension.
    assert not dist.is_initialized()
    dist.init_process_group("fake", store=FakeStore(), rank=1, world_size=4)
    try:
        yield DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("edp_shard", "ep"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "module,recipe,profile_factory",
    [
        ("torchtitan_npu_recipes.tests.models.deepseek_v4", "deepseek_v4_debugmodel", _dsv4_muon_profile),
        ("torchtitan_npu_recipes.models.deepseek_v4_1", "deepseek_v4_1_debugmodel_text", _v41_muon_profile),
    ],
    ids=["v4", "v4.1"],
)
@pytest.mark.parametrize(
    "projection,global_shape,local_shape,row_axis",
    [("w13", (2, 2, 8, 4), (1, 2, 4, 4), 2), ("w2", (2, 4, 8), (1, 2, 8), 1)],
    ids=["stacked-gate-up", "down"],
)
def test_recipe_expert_layout_preserves_ep_ownership_during_row_redistribution(
    sparse_storage_mesh, module, recipe, profile_factory, projection, global_shape, local_shape, row_axis
):
    config = getattr(importlib.import_module(module), recipe)()
    profile = profile_factory(config.model)
    config.optim.optimizer = OptimizerConfig(
        optimizers=[
            DistMuon.Config(
                pattern=profile.muon_pattern,
                compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
                bucket_configs=profile.bucket_configs,
                lr=1e-5,
            ),
            AdamW.Config(pattern=r".*"),
        ]
    )
    fqn = f"layers.0.feed_forward.routed_experts.{projection}.weight"
    optimizer_config = next(
        owner for owner in config.optim.optimizer.optimizers if hasattr(owner, "compute_sharding_by_fqn")
    )
    layout = optimizer_config.compute_sharding_by_fqn[fqn]
    local = torch.zeros(local_shape)
    parameter = torch.nn.Parameter(
        DTensor.from_local(
            local,
            sparse_storage_mesh,
            (Shard(row_axis), Shard(0)),
            shape=torch.Size(global_shape),
            stride=torch.empty(global_shape).stride(),
            run_check=False,
        )
    )

    resolved = _resolve_storage_to_compute_transition(fqn, parameter, layout)

    # EP keeps its expert subset; only EDP gathers matrix rows and redistributes
    # whole expert matrices within that subset. Reversing this order would mix
    # expert ownership across the EP boundary.
    assert resolved.redistribution_storage_mesh_axis == 0
    assert resolved.compute_sharding == Shard(0)
    assert resolved.resolved_compute_layout_signature == (
        ("edp_shard", _StridedShard(0, split_factor=2)),
        ("ep", Shard(0)),
    )
    assert parameter.placements == (Shard(row_axis), Shard(0))
    assert parameter.to_local().shape == local_shape
    with pytest.raises(ValueError, match="must declare shard_order_by_tensor_dim"):
        _resolve_storage_to_compute_transition(fqn, parameter, replace(layout, shard_order_by_tensor_dim={}))
