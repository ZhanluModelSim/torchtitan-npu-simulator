# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import fnmatch

import pytest
import torch
from torch.distributed.tensor import Shard
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.distributed.flex_shard import BlockShard, Owned

from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
from torchtitan_npu_recipes.models.deepseek_v4 import _dsv4_muon_profile
from torchtitan_npu_recipes.tests.ci.deepseek_v4 import deepseek_v4_flash_lora
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import (
    _v41_muon_profile,
    deepseek_v4_1_flash_40layers_16experts_text_lora,
)


@pytest.mark.parametrize(
    "factory,profile_factory",
    [
        (deepseek_v4_flash_lora, _dsv4_muon_profile),
        (deepseek_v4_1_flash_40layers_16experts_text_lora, _v41_muon_profile),
    ],
    ids=["v4", "v4.1"],
)
def test_actual_lora_recipe_muon_groups_cover_adapters_with_layouts_and_unique_buckets(factory, profile_factory):
    config = factory()
    optimizer = config.optim.optimizer
    assert [type(group) for group in optimizer.optimizers] == [AdamW.Config]
    with torch.device("meta"):
        model = config.model.build()
    trainables = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainables and all(is_lora_parameter(name) for name in trainables)
    assert not any("indexer" in name or "engram" in name or "vision" in name for name in trainables)

    profile = profile_factory(config.model)
    optimizer = OptimizerConfig(
        optimizers=[
            DistMuon.Config(
                pattern=profile.muon_pattern,
                compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
                bucket_configs=profile.bucket_configs,
                lr=1e-5,
            ),
            AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8),
        ]
    )
    claimed = set()
    groups = [OptimizersContainer._build_param_group(model, group, claimed) for group in optimizer.optimizers]
    assert claimed == trainables
    names = []
    for group in groups:
        names.extend(group["param_names"])
    assert len(names) == len(set(names)) == len(trainables)
    assert isinstance(optimizer.optimizers[0], DistMuon.Config)
    muon = optimizer.optimizers[0]
    muon_names = set(groups[0]["param_names"])
    adamw_names = set(groups[1]["param_names"])
    assert adamw_names == {"lm_head.lora_a.weight", "lm_head.lora_b.weight"}
    assert muon_names == trainables - adamw_names
    for name in muon_names:
        assert name in muon.compute_sharding_by_fqn
        matches = [
            bucket
            for bucket in muon.bucket_configs
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in bucket.patterns)
        ]
        assert len(matches) == 1, name

    for factor in ("a", "b"):
        dense = f"layers.0.attention.wq_a.lora_{factor}.weight"
        grouped = f"layers.0.feed_forward.routed_experts.w13.lora_{factor}.weight"
        assert {dense, grouped} <= muon_names
        expert_layout = muon.compute_sharding_by_fqn[grouped].shardings_by_mesh_axis
        assert {"dp_shard", "edp_shard", "ep"} <= expert_layout.keys()
        assert all(isinstance(placement, Shard) and placement.dim == 0 for placement in expert_layout.values())
        assert all(
            isinstance(placement, Owned)
            for placement in muon.compute_sharding_by_fqn[dense].shardings_by_mesh_axis.values()
        )

    for projection in ("wq_b", "wo_a"):
        prefix = f"layers.0.attention.{projection}"
        a_layout = muon.compute_sharding_by_fqn[f"{prefix}.lora_a.weight"]
        b_layout = muon.compute_sharding_by_fqn[f"{prefix}.lora_b.weight"]
        assert all(isinstance(placement, Owned) for placement in a_layout.shardings_by_mesh_axis.values())
        assert b_layout == muon.compute_sharding_by_fqn[f"{prefix}.weight"]
        assert all(isinstance(placement, BlockShard) for placement in b_layout.shardings_by_mesh_axis.values())
