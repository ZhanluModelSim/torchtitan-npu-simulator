# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Model-owned sharding declarations reach native DTensor storage consumers."""

import importlib
import threading

import pytest
import torch
import torch.distributed as dist
import torchtitan.distributed.parallelism_context as context_module
import torchtitan.distributed.spmd_types as spmd_runtime
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.debug import _clear_sharding_prop_cache
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.parallelism_context import DistributedTopology, ParallelismContext


@pytest.fixture
def cpu_storage_world(monkeypatch):
    # Fake PG supplies mesh metadata only. No assertion below claims collective
    # or NPU correctness; state distribution and Module recursion remain real.
    assert not dist.is_initialized()
    # nn.Parameter's DTensor detach can reuse a cached output spec from an
    # equal-valued mesh owned by a previous process group. Clear both Python
    # and C++ propagation caches, whose specs retain the old mesh identity.
    _clear_sharding_prop_cache()
    dist.init_process_group("fake", store=FakeStore(), rank=1, world_size=2)
    monkeypatch.setattr(context_module, "device_type", "cpu")
    # activate_spmd registers meshes beyond its lexical scope. Isolate that
    # registry so the fixture cannot leave destroyed process groups behind.
    monkeypatch.setattr(spmd_runtime, "_MESH_TLS", threading.local())
    try:
        yield
    finally:
        _clear_sharding_prop_cache()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "family,recipe",
    [("deepseek_v4", "deepseek_v4_debugmodel"), ("deepseek_v4_1", "deepseek_v4_1_debugmodel_text")],
    ids=["v4", "v4.1"],
)
@pytest.mark.parametrize("ep", [1, 2], ids=["dense-experts", "expert-parallel"])
@pytest.mark.parametrize("full_ac", [False, True], ids=["no-ac", "full-ac"])
def test_model_parallelize_places_expert_weights_on_dense_or_sparse_storage(
    cpu_storage_world, family, recipe, ep, full_ac
):
    registry = importlib.import_module(f"torchtitan_npu.models.{family}.config_registry")
    config = getattr(registry, recipe)()
    config.hf_assets_path = "/tmp"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = ep
    # One real block is sufficient to exercise the model's sharding policy and
    # native recursive state distribution, without allocating model weights.
    config.model.layers = config.model.layers[:1]
    config.model.n_layers = 1
    config.model.compress_ratios = config.model.compress_ratios[:1]
    if family == "deepseek_v4_1":
        config.model.kv_source_layers = config.model.kv_source_layers[:1]
        config.model.layers[0].engram = None
    config.model.update_from_config(config=config)
    context = ParallelismContext.from_config(config.parallelism, DistributedTopology(world_size=2))
    context.build_mesh()
    with torch.device("meta"):
        model = config.model.build()
    experts = model.layers["0"].moe.routed_experts
    shapes = {name: tuple(getattr(experts, name).weight.shape) for name in ("w13", "w2")}

    # Match the production lifecycle: _parallelize materializes plain local
    # tensors with SPMD annotations; FSDP consumes those annotations to create
    # DTensor storage. Neither operation is replaced with a test double.
    model.parallelize(
        parallelism_context=context,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=None,
        ac_config=FullAC.Config() if full_ac else None,
        dump_folder="",
    )

    assert isinstance(model.layers["0"], CheckpointWrapper) is full_ac

    # There are at least four experts, so these two ranks split the expert
    # dimension. EP1 assigns the split to DP; EP2 retains sparse EP ownership
    # with an explicitly retained size-one EDP shard axis. Since EP owns its
    # expert subset before FSDP partitions it, EDP's repeated dim-0 shard is
    # strided by the two EP ranks even though this EDP axis has size one.
    expected_axes = ("dp_shard",) if ep == 1 else ("edp_shard", "ep")
    expected_placements = (Shard(0),) if ep == 1 else (_StridedShard(0, split_factor=2), Shard(0))
    for name, global_shape in shapes.items():
        weight = getattr(experts, name).weight
        assert isinstance(weight, DTensor)
        assert weight.device_mesh.mesh_dim_names == expected_axes
        assert weight.placements == expected_placements
        assert tuple(weight.shape) == global_shape
        assert tuple(weight.to_local().shape) == (global_shape[0] // 2, *global_shape[1:])
        assert weight.requires_grad


@pytest.mark.parametrize("full_ac", [False, True], ids=["no-ac", "full-ac"])
def test_multimodal_parallelize_shards_all_vision_parameters(cpu_storage_world, full_ac):
    from torchtitan_npu.models.deepseek_v4_1.config_registry import deepseek_v4_1_debugmodel_multimodal

    config = deepseek_v4_1_debugmodel_multimodal()
    config.hf_assets_path = "/tmp"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 1
    config.model.layers = config.model.layers[:1]
    config.model.n_layers = 1
    config.model.compress_ratios = config.model.compress_ratios[:1]
    config.model.kv_source_layers = config.model.kv_source_layers[:1]
    config.model.layers[0].engram = None
    config.model.vision_encoder.num_layers = 1
    config.model.update_from_config(config=config)
    context = ParallelismContext.from_config(config.parallelism, DistributedTopology(world_size=2))
    context.build_mesh()
    with torch.device("meta"):
        model = config.model.build()
    vision_parameters = list(model.vision_encoder.parameters())
    shapes = [tuple(parameter.shape) for parameter in vision_parameters]

    model.parallelize(
        parallelism_context=context,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=None,
        ac_config=FullAC.Config() if full_ac else None,
        dump_folder="",
    )

    assert isinstance(model.vision_encoder.blocks[0], CheckpointWrapper) is full_ac
    # FSDP must partition every projection, bias and norm, including the patch
    # embedding and aligner outside the checkpointed transformer blocks.
    for parameter, shape in zip(model.vision_encoder.parameters(), shapes, strict=True):
        assert isinstance(parameter, DTensor)
        assert parameter.device_mesh.mesh_dim_names == ("dp_shard",)
        assert parameter.placements == (Shard(0),)
        assert tuple(parameter.shape) == shape
        assert tuple(parameter.to_local().shape) == (shape[0] // 2, *shape[1:])
        assert parameter.requires_grad
