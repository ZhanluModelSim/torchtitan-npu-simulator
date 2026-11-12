# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Muon optimizer selection and layouts for target-version DeepSeek models."""

import re

import pytest
import torch
from torch.distributed.tensor import Shard
from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.components.optim.optimizer import OptimizersContainer
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName

from torchtitan_npu.models.deepseek_v4 import model_registry as v4_registry
from torchtitan_npu.models.deepseek_v4.muon import _dsv4_optimizer_config
from torchtitan_npu.models.deepseek_v4_1 import model_registry as v41_registry
from torchtitan_npu.models.deepseek_v4_1.optimizer import _v41_optimizer_config


@pytest.mark.parametrize("version", ["v4", "v41"])
def test_muon_groups_match_real_parameters_layouts_and_buckets(version):
    if version == "v4":
        config = v4_registry("debugmodel", seq_len=128, num_mtp_layers=1)
        optimizer_config = _dsv4_optimizer_config(config, lr=1e-5)
    else:
        config = v41_registry("deepseek_v4_1_debugmodel_text")
        optimizer_config = _v41_optimizer_config(config)
    if version == "v41":
        assert optimizer_config.name == "AdamW"
        # v0.4.0 keeps the recipe-owned groups in ``_optimizer_defaults`` until
        # ``materialize()`` selects a mode.
        sparse = optimizer_config._optimizer_defaults["AdamW"][0]
    optimizer_config.name = "Muon"
    optimizer_config.materialize()
    if version == "v41":
        assert len(optimizer_config.optimizers) == 3 and optimizer_config.optimizers[0] is sparse
        optimizer_config.materialize()
        assert len(optimizer_config.optimizers) == 3 and optimizer_config.optimizers[0] is sparse
    profile = optimizer_config._muon_profile
    assert profile is not None
    with torch.device("meta"):
        model = config.build()

    groups = optimizer_config.optimizers
    muon = next(group for group in groups if isinstance(group, DistMuon.Config))
    adamw = next(group for group in groups if isinstance(group, AdamW.Config))
    claimed = set()
    assigned = []
    for group in groups:
        result = OptimizersContainer._build_param_group(model, group, claimed)
        assigned.append((group, set(result["param_names"])))
    muon_names = next(names for group, names in assigned if group is muon)
    adamw_names = next(names for group, names in assigned if group is adamw)
    all_names = {name for name, param in model.named_parameters() if param.requires_grad}
    layouts = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]
    buckets = profile.optimizer_factory_kwargs["DistMuon"]["bucket_configs"]
    bucket_names = []
    for bucket in buckets:
        bucket_names.extend(bucket.patterns)
    assert set(bucket_names) == layouts.keys() == muon_names
    assert all(layout.compute_dtype is torch.bfloat16 for layout in muon.compute_sharding_by_fqn.values())
    assert len(bucket_names) == len(set(bucket_names))
    assert claimed == all_names and muon_names.isdisjoint(adamw_names)
    assert re.search(profile.muon_pattern, "layers.0.feed_forward.shared_experts.w13.weight")
    for name in (
        "layers.0.feed_forward.shared_experts.w13.weight",
        "layers.0.feed_forward.shared_experts.w2.weight",
        "layers.0.feed_forward.routed_experts.w13.weight",
        "layers.0.feed_forward.routed_experts.w2.weight",
        "layers.0.feed_forward.router.gate.weight",
    ):
        assert name in muon_names
    if version == "v4":
        assert "mtp_layers.0.feed_forward.routed_experts.w13.weight" in muon_names
        assert "mtp_layers.0.feed_forward.router.gate.weight" in muon_names
        for suffix in ("ape", "wkv.weight", "wgate.weight"):
            assert f"layers.3.attention.compressor_128.{suffix}" in muon_names
        # Native Linear.out_features flattens the groups; Muon must still
        # orthogonalize each [o_lora_rank, per_group_input] matrix separately.
        attention = config.layers[0].attention
        assert attention.wo_a.out_features == attention.n_groups * attention.o_lora_rank
    else:
        assert "layers.1.engram.table.weight" not in muon_names | adamw_names
        assert "layers.1.engram.gate.wkv" in muon_names
        assert "layers.1.engram.gate.q_weight" in adamw_names
        assert any(
            "engram.table.weight" in name
            for group, names in assigned
            if group is not muon and group is not adamw
            for name in names
        )
    indexer_group = muon_names if version == "v4" else adamw_names
    assert any(name.endswith(".attention.indexer.wq_b.weight") for name in indexer_group)

    dp = MeshAxisName.DP_SHARD.value
    q_shard = layouts["layers.0.attention.wq_b.weight"].shardings_by_mesh_axis[dp]
    assert isinstance(q_shard, BlockShard) and q_shard.dim == 0
    assert q_shard.block_sizes == (config.layers[0].attention.head_dim,)
    out_shard = layouts["layers.0.attention.wo_a.weight"].shardings_by_mesh_axis[dp]
    assert isinstance(out_shard, BlockShard) and out_shard.block_sizes == (
        config.layers[0].attention.o_lora_rank if version == "v4" else config.layers[0].attention.wo_a.out_features,
    )
    for name in ("layers.0.feed_forward.router.gate.weight", "layers.0.attention.wq_a.weight"):
        assert isinstance(layouts[name].shardings_by_mesh_axis[dp], Owned)
    expert = layouts["layers.0.feed_forward.routed_experts.w13.weight"]
    assert set(expert.shardings_by_mesh_axis) == {dp, MeshAxisName.EDP_SHARD.value, MeshAxisName.EP.value}
    assert all(isinstance(shard, Shard) and shard.dim == 0 for shard in expert.shardings_by_mesh_axis.values())
