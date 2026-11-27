# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation and real FSDP2 DTensor layout checks for disable_expert_replication."""

import sys
import threading
import types
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torchtitan.distributed.parallelism_context as context_module
import torchtitan.distributed.spmd_types as spmd_runtime
from torch.distributed.fsdp import DataParallelMeshDims
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.debug import _clear_sharding_prop_cache
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan.config import ConfigLoader
from torchtitan.distributed.flex_shard.dist_muon import (
    _NoRedistributionTransition,
    _resolve_storage_to_compute_transition,
)
from torchtitan.distributed.parallelism_context import DistributedTopology, ParallelismContext
from torchtitan.trainer import Trainer

from torchtitan_npu.config import OptimConfig, OptimizerConfig, ParallelismConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.fsdp import disable_expert_replication_dims
from torchtitan_npu.models.common.muon import make_expert_layout


def _load_recipe(monkeypatch, factory, *, name: str = "recipe"):
    module_name = f"_torchtitan_npu_expert_replication_{name}"
    registry = types.ModuleType(module_name)
    setattr(registry, name, factory)
    monkeypatch.setitem(sys.modules, module_name, registry)
    return ConfigLoader().load(["--module", module_name, "--config", name])


def _hsdp(*, enabled: bool) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_shard_degree=4,
        data_parallel_replicate_degree=2,
        expert_parallel_degree=4,
        disable_expert_replication=enabled,
    )


@pytest.fixture
def cpu_world_8p(monkeypatch):
    assert not dist.is_initialized()
    _clear_sharding_prop_cache()
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
    monkeypatch.setattr(context_module, "device_type", "cpu")
    monkeypatch.setattr(spmd_runtime, "_MESH_TLS", threading.local())
    try:
        yield
    finally:
        _clear_sharding_prop_cache()
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["v4", "v4.1"])
@pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])
def test_parallelize_shards_experts_across_replicas(cpu_world_8p, monkeypatch, tmp_path, family, enabled):
    from torchtitan_npu_recipes.models.deepseek_v4 import _flash_config
    from torchtitan_npu_recipes.models.deepseek_v4_1 import deepseek_v4_1_debugmodel_text

    parallelism = _hsdp(enabled=enabled)

    def make_recipe():
        if family == "v4":
            return _flash_config(
                "deepseek_v4_flash_43layers_16experts",
                parallelism=parallelism,
                seq_len=512,
                num_layers=1,
                num_mtp_layers=1,
                hf_assets_path=str(tmp_path),
            )
        cfg = deepseek_v4_1_debugmodel_text(parallelism=parallelism, hf_assets_path=str(tmp_path))
        cfg.model.layers = cfg.model.layers[:1]
        cfg.model.n_layers = 1
        cfg.model.compress_ratios = cfg.model.compress_ratios[:1]
        cfg.model.kv_source_layers = cfg.model.kv_source_layers[:1]
        cfg.model.layers[0].engram = None
        return cfg

    config = _load_recipe(monkeypatch, make_recipe, name=f"{family}_{enabled}")
    context = ParallelismContext.from_config(config.parallelism, DistributedTopology(world_size=8), dump_folder="")
    context.build_mesh()
    with torch.device("meta"):
        model = config.model.build()
    block = model.layers["0"]
    dense_shape = tuple(block.attention.wq_a.weight.shape)
    expert_w2_shape = tuple(block.feed_forward.routed_experts.w2.weight.shape)

    model.parallelize(
        parallelism_context=context,
        training=config.training,
        parallelism=config.parallelism,
        local_compile_regions=config.model.local_compile_regions,
        activation_remat=None,
        dump_folder="",
    )

    dense = block.attention.wq_a.weight
    assert isinstance(dense, DTensor)
    assert dense.device_mesh.mesh_dim_names == ("dp_replicate", "dp_shard")
    assert dense.placements == (Replicate(), Shard(0))
    assert tuple(dense.to_local().shape) == (dense_shape[0] // 4, *dense_shape[1:])

    blocks = [block] + (list(model.mtp_layers) if getattr(model, "mtp_layers", None) else [])
    expert_layout = make_expert_layout(("dp_shard",))
    for target in blocks:
        w2 = target.feed_forward.routed_experts.w2.weight
        assert isinstance(w2, DTensor)
        if not enabled:
            assert w2.device_mesh.mesh_dim_names == ("dp_replicate", "edp_shard", "ep")
            assert w2.placements == (Replicate(), _StridedShard(0, split_factor=4), Shard(0))
            assert tuple(w2.to_local().shape) == (expert_w2_shape[0] // 4, *expert_w2_shape[1:])
        else:
            assert w2.device_mesh.mesh_dim_names == ("dp_replicate_edp_shard", "ep")
            assert w2.placements == (_StridedShard(0, split_factor=4), Shard(0))
            assert tuple(w2.to_local().shape) == (expert_w2_shape[0] // 8, *expert_w2_shape[1:])
            resolved = _resolve_storage_to_compute_transition("w2.weight", w2, expert_layout)
            assert isinstance(resolved.storage_to_compute_transition, _NoRedistributionTransition)


def test_rejects_effective_expert_dp_exceeding_num_experts(cpu_world_8p, monkeypatch, tmp_path):
    from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

    config = _load_recipe(
        monkeypatch,
        lambda: deepseek_v4_debugmodel(parallelism=_hsdp(enabled=True), hf_assets_path=str(tmp_path)),
        name="v4_4e",
    )
    config.model.layers = config.model.layers[:1]
    config.model.n_layers = 1
    config.model.compress_ratios = config.model.compress_ratios[:1]
    context = ParallelismContext.from_config(config.parallelism, DistributedTopology(world_size=8), dump_folder="")
    context.build_mesh()
    with torch.device("meta"):
        model = config.model.build()

    with pytest.raises(ValueError, match=r"effective expert DP \(8\) <= num_experts \(4\)"):
        model.parallelize(
            parallelism_context=context,
            training=config.training,
            parallelism=config.parallelism,
            local_compile_regions=config.model.local_compile_regions,
            activation_remat=None,
            dump_folder="",
        )


@pytest.mark.parametrize("shard_over_efsdp", [False, True], ids=["ep-only", "joint-efsdp"])
def test_v41_engram_sharding_with_disable_expert_replication(cpu_world_8p, monkeypatch, tmp_path, shard_over_efsdp):
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable
    from torchtitan_npu_recipes.models.deepseek_v4_1 import deepseek_v4_1_debugmodel_text

    def make_recipe():
        cfg = deepseek_v4_1_debugmodel_text(parallelism=_hsdp(enabled=True), hf_assets_path=str(tmp_path))
        engram_layer = next(layer for layer in cfg.model.layers if layer.engram is not None)
        engram_layer.engram.table.shard_over_efsdp = shard_over_efsdp
        cfg.model.layers = [engram_layer]
        cfg.model.n_layers = 1
        cfg.model.compress_ratios = cfg.model.compress_ratios[:1]
        cfg.model.kv_source_layers = cfg.model.kv_source_layers[:1]
        return cfg

    config = _load_recipe(monkeypatch, make_recipe, name=f"v41_engram_{shard_over_efsdp}")
    context = ParallelismContext.from_config(config.parallelism, DistributedTopology(world_size=8), dump_folder="")
    context.build_mesh()
    with torch.device("meta"):
        model = config.model.build()
    table = next(m for m in model.modules() if isinstance(m, HostEngramTable))
    total_rows = table.num_embeddings

    model.parallelize(
        parallelism_context=context,
        training=config.training,
        parallelism=config.parallelism,
        local_compile_regions=config.model.local_compile_regions,
        activation_remat=None,
        dump_folder="",
    )

    assert not isinstance(table.weight, DTensor)
    if shard_over_efsdp:
        assert table._efsdp_size == 2
        assert table.lookup_mesh.size() == 8
        assert table.weight.shape[0] == total_rows // 8
        assert table._replica_size == 1
        assert table._replica_group is None
    else:
        assert table._efsdp_size == 1
        assert table.lookup_mesh.size() == 4
        assert table.weight.shape[0] == total_rows // 4
        assert table._replica_size == 2
        assert table._replica_group is not None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"data_parallel_replicate_degree": 1}, "data_parallel_replicate_degree > 1"),
        ({"expert_parallel_degree": 1}, "expert_parallel_degree > 1"),
        ({"tensor_parallel_degree": 2}, "TP=CP=PP=1"),
        ({"context_parallel_degree": 2}, "TP=CP=PP=1"),
        ({"pipeline_parallel_degree": 2}, "TP=CP=PP=1"),
    ],
)
def test_parallelism_rejects_unsupported_degrees(kwargs, match):
    values = {"data_parallel_replicate_degree": 2, "expert_parallel_degree": 4, **kwargs}
    with pytest.raises(ValueError, match=match):
        ParallelismConfig(disable_expert_replication=True, **values)


def test_trainer_and_loader_reject_unsupported_combinations(monkeypatch, tmp_path):
    from torchtitan_npu.models.deepseek_v4_1.model import (
        GraphTrainerDeepSeekV41Model,
        GraphTrainerDeepSeekV41MultimodalModel,
    )
    from torchtitan_npu_recipes.graph_trainer.deepseek_v4 import graph_trainer_deepseek_v4_debugmodel
    from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

    default_cfg = TrainerEx.Config(model=None)
    assert isinstance(default_cfg.parallelism, ParallelismConfig)
    assert default_cfg.parallelism.disable_expert_replication is False
    assert GraphTrainerDeepSeekV41Model.Config._supports_disable_expert_replication is False
    assert GraphTrainerDeepSeekV41MultimodalModel.Config._supports_disable_expert_replication is False

    with pytest.raises(ValueError, match="not supported by GraphTrainer"):
        _load_recipe(
            monkeypatch,
            lambda: replace(graph_trainer_deepseek_v4_debugmodel(), parallelism=_hsdp(enabled=True)),
            name="graph_v4",
        )

    with pytest.raises(ValueError, match=r"requires DeepSeek-V4 or DeepSeek-V4\.1"):
        _load_recipe(
            monkeypatch,
            lambda: Trainer.Config(model=None, hf_assets_path=str(tmp_path), parallelism=_hsdp(enabled=True)),
            name="unsupported_model",
        )

    base = deepseek_v4_debugmodel(hf_assets_path=str(tmp_path))
    with pytest.raises(ValueError, match="incompatible with enable_hsdp_replica_dedup"):
        TrainerEx.Config(
            model=base.model,
            optim=OptimConfig(
                optimizer=OptimizerConfig(
                    optimizers=[
                        DistMuon.Config(
                            pattern=r".*",
                            compute_sharding_by_fqn={},
                            bucket_configs=(),
                            enable_hsdp_replica_dedup=True,
                        )
                    ]
                )
            ),
            parallelism=_hsdp(enabled=True),
        )

    mutated = TrainerEx.Config(model=base.model, hf_assets_path=str(tmp_path))
    mutated.parallelism.disable_expert_replication = True
    with pytest.raises(ValueError, match="data_parallel_replicate_degree > 1"):
        TrainerEx(mutated)

    with pytest.raises(ValueError, match="replicate axis"):
        disable_expert_replication_dims(
            torch.nn.Module(), {}, DataParallelMeshDims(shard="edp_shard", replicate=None), ep_degree=4
        )
