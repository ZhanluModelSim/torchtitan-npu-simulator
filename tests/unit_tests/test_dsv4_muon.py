# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
from pathlib import Path

from torch.distributed.tensor import Shard
from torchtitan.distributed.flex_shard import BlockShard
from torchtitan.distributed.parallelism_context import MeshAxisName

from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4.muon import _dsv4_muon_profile

_ROOT = Path(__file__).resolve().parents[2]


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(path.read_text(), node) or ""
    raise AssertionError(f"function {name!r} not found in {path}")


def test_d2h_swap_cleanup_warns_before_releasing_the_pinned_buffer():
    path = _ROOT / "torchtitan_npu/extensions/novaswap/swap_primitive.py"
    source = _function_source(path, "d2h")

    assert "logger.warning" in source
    assert "D2H swap submission failed" in source
    assert "stream.synchronize()" in source
    assert "PinnedCpuStorage.free(tensor_cpu, owner=owner)" in source
    assert "raise" in source


def test_dsv4_muon_profile_uses_validated_phase_one_policy():
    from torchtitan.components.optim import AdamW

    from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
    from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()
    optimizer = config.optim.optimizer
    assert optimizer.name == "AdamW"
    assert len(optimizer.optimizers) == 1 and isinstance(optimizer.optimizers[0], AdamW.Config)
    optimizer.name = "Muon"
    optimizer.materialize()
    muon, adamw = optimizer.optimizers
    assert isinstance(muon, DistMuon.Config) and isinstance(adamw, AdamW.Config)
    assert muon.momentum == 0.95 and muon.ns_steps == 10
    assert adamw.betas == (0.9, 0.95) and adamw.eps == 1e-8
    assert muon.pattern == optimizer._muon_profile.muon_pattern


def test_dsv4_unified_muon_policy_follows_paper_parameter_split():
    path = _ROOT / "torchtitan_npu/models/deepseek_v4/muon.py"
    source = path.read_text()
    active_source = _function_source(path, "_dsv4_muon_profile")

    assert "mtp_layers" in active_source
    assert "_PAPER_" not in source
    assert "_distributed_paper_parameter_muon_optimizer" not in source
    assert "deepseek_v4_debugmodel_paper_muon" not in source
    assert "BlockShard(dim=0" in source

    profile = _dsv4_muon_profile(model_registry("debugmodel"))
    compute_shardings = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]
    expert_layout = compute_shardings["layers.2.feed_forward.routed_experts.w13.weight"]
    assert all(isinstance(p, Shard) and p.dim == 0 for p in expert_layout.shardings_by_mesh_axis.values())
    assert set(expert_layout.shardings_by_mesh_axis) == {
        MeshAxisName.DP_SHARD.value,
        MeshAxisName.EDP_SHARD.value,
        MeshAxisName.EP.value,
    }
    assert dict(expert_layout.shard_order_by_tensor_dim) == {
        0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
    }
    bucket_configs = profile.optimizer_factory_kwargs["DistMuon"]["bucket_configs"]
    routed_expert_bucket = next(bucket for bucket in bucket_configs if bucket.name == "layers.2.routed-experts")
    assert all(".feed_forward.routed_experts." in fqn for fqn in routed_expert_bucket.patterns)
    for unified_muon_parameter in (
        "compressor",
        "routed_experts",
        "router",
        'hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")',
        'mtp_projections = ("e_proj", "h_proj")',
    ):
        assert unified_muon_parameter in active_source


def test_dsv4_muon_indexer_query_uses_per_head_block_sharding():
    model_spec = model_registry("debugmodel")
    profile = _dsv4_muon_profile(model_spec)
    compute_shardings = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]

    main_query_layout = compute_shardings["layers.2.attention.wq_b.weight"]
    indexer_query_layout = compute_shardings["layers.2.attention.indexer.wq_b.weight"]

    main_dp_sharding = main_query_layout.shardings_by_mesh_axis["dp_shard"]
    assert isinstance(main_dp_sharding, BlockShard)
    assert main_dp_sharding.dim == 0
    assert main_dp_sharding.block_sizes == (model_spec.layers[2].attention.head_dim,)

    indexer_dp_sharding = indexer_query_layout.shardings_by_mesh_axis["dp_shard"]
    assert isinstance(indexer_dp_sharding, BlockShard)
    assert indexer_dp_sharding.dim == 0
    indexer = model_spec.layers[2].attention.indexer
    assert indexer is not None
    assert indexer_dp_sharding.block_sizes == (indexer.index_head_dim,)


def test_graph_trainer_conversion_strips_only_npu_top_level_extension():
    conversion_path = _ROOT / "torchtitan_npu/models/deepseek_v4/graph_trainer.py"
    registry_path = _ROOT / "torchtitan_npu_recipes/graph_trainer/deepseek_v4.py"
    conversion_source = _function_source(conversion_path, "to_graph_trainer_config")

    assert "isinstance(base_config, TrainerEx.Config)" in conversion_source
    assert "for config_field in fields(Trainer.Config)" in conversion_source

    for graph_recipe in (
        "graph_trainer_deepseek_v4_debugmodel",
        "graph_trainer_deepseek_v4_flash",
        "graph_trainer_deepseek_v4_flash_43layers_16experts",
        "graph_trainer_deepseek_v4_pro",
        "graph_trainer_deepseek_v4_pro_61layers_32experts",
    ):
        assert "use_npu_config" not in _function_source(registry_path, graph_recipe)
