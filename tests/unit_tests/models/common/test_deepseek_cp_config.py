# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from types import SimpleNamespace

import pytest
import torch
from torchtitan.config import ConfigManager
from torchtitan.config.override import apply_overrides
from torchtitan.models.common.cp_attention import CPInnerAttention

_RECIPES = [
    ("deepseek_v4", "deepseek_v4_flash", "deepseek_v4_flash_cp", "AscV4ContextParallelSparseAttention", 128),
    (
        "deepseek_v4_1",
        "deepseek_v4_1_flash_text",
        "deepseek_v4_1_flash_text_cp",
        "AscV41ContextParallelSparseAttention",
        2,
    ),
]


def _selectors(config, family):
    if family == "deepseek_v4":
        return [layer.attention.lightning_indexer for layer in config.model.layers if layer.attention.lightning_indexer]
    return [layer.attention.indexer.selector for layer in config.model.layers]


@pytest.mark.parametrize("family,original,recipe,owner_name,alignment", _RECIPES)
def test_original_cp1_recipe_keeps_ordinary_attention(family, original, recipe, owner_name, alignment):
    registry = importlib.import_module(f"torchtitan_npu.models.{family}.config_registry")
    config = getattr(registry, original)()
    assert config.parallelism.context_parallel_degree == 1
    assert all(
        not isinstance(layer.attention.inner_attention, CPInnerAttention.Config) for layer in config.model.layers
    )


@pytest.mark.parametrize("cp", [2, 128])
@pytest.mark.parametrize("family,original,recipe,owner_name,alignment", _RECIPES)
def test_typed_cp_recipe_passes_native_cli_validation_and_retains_fused_owner(
    monkeypatch, cp, family, original, recipe, owner_name, alignment
):
    monkeypatch.setenv(
        "DSV4_PER_DOC_ALIGNMENT" if family == "deepseek_v4" else "DSV41_PER_DOC_ALIGNMENT", str(alignment * cp)
    )
    config = ConfigManager().parse_args(
        [
            "--module",
            f"torchtitan_npu.models.{family}.config_registry",
            "--config",
            recipe,
            "--hf-assets-path",
            "/tmp",
            "--parallelism.context-parallel-degree",
            str(cp),
            "--training.max-context-length",
            "1048576",
            "--training.num-tokens-per-microbatch-per-dp-rank",
            "1048576",
        ]
    )
    core_module = importlib.import_module(f"torchtitan_npu.override.{family}.sparse_attn.ascendc")
    owner = getattr(core_module, owner_name)
    fused_owner = owner.__bases__[1]
    # A normal exact core override remains harmless after explicit CP selection.
    config.override.imports.append(f"torchtitan_npu.override.{family}.sparse_attn.asc")
    config.model.update_from_config(config=config)
    apply_overrides(config.override, config)
    cores = [layer.attention.inner_attention for layer in config.model.layers]
    selector = _selectors(config, family)[0]
    selector_owner = selector._owner
    assert selector_owner.__module__.startswith(f"torchtitan_npu.override.{family}.lightning_indexer")
    assert config.parallelism.context_parallel_degree == cp
    assert all(isinstance(core, CPInnerAttention.Config) and core._owner is owner for core in cores)
    assert all(core.num_heads == 64 and core.head_dim == 512 for core in cores)
    with torch.device("meta"):
        built = cores[0].build()
    assert isinstance(built, CPInnerAttention)
    assert type(built).forward is fused_owner.forward
    config.__post_init__()
    apply_overrides(config.override, config)
    assert all(layer.attention.inner_attention._owner is owner for layer in config.model.layers)
    selector = _selectors(config, family)[0]
    assert selector._owner is selector_owner
    config.parallelism.context_parallel_degree = 1
    with pytest.raises(ValueError, match=r"CPInnerAttention.*degree is 1"):
        config.__post_init__()


@pytest.mark.parametrize("family,original,recipe,owner_name,alignment", _RECIPES)
def test_typed_cp_metadata_hook_rejects_reference_metadata_and_reaches_real_builder(
    monkeypatch, family, original, recipe, owner_name, alignment
):
    from torchtitan.distributed import context_parallel as native_cp
    from torchtitan.models.common.decoder_sharding import decoder_input_sharding

    from torchtitan_npu.patches.torchtitan.distributed import context_parallel as doc_cp
    from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context

    module = importlib.import_module(f"torchtitan_npu.override.{family}.sparse_attn.ascendc")
    owner = getattr(module, owner_name)
    metadata_owner = module.AscV4Metadata if family == "deepseek_v4" else module.AscV41Metadata
    with pytest.raises(TypeError, match="requires AscV"):
        owner.prepare_cp_metadata({"layer": object()}, permutation=None)

    registry = importlib.import_module(f"torchtitan_npu.models.{family}.config_registry")
    config = getattr(registry, recipe)()
    config.model.layers = config.model.layers[:1]
    config.model.n_layers = 1
    config.model.compress_ratios = config.model.compress_ratios[:1]
    with torch.device("meta"):
        model = config.model.build()
    positions = torch.arange(8)
    core = config.model.layers[0].attention.inner_attention
    metadata = core.build_attention_metadata(positions)
    assert isinstance(metadata, metadata_owner)
    with pytest.raises(RuntimeError, match=r"only Decoder\._cp_shard publishes"):
        owner.prepare_cp_metadata({"layer": metadata}, permutation=None)

    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
    monkeypatch.setattr(native_cp, "spmd_mesh_group", lambda axis: mesh)
    monkeypatch.setattr(doc_cp, "spmd_mesh_group", lambda axis: mesh)

    def local_shard(tensor, group, *, src, dst):
        # Only the collective is local: the real producer, metadata builders,
        # load-balancer permutation and shard_tensors input handling all run.
        return tensor.chunk(group.size(), dim=0)[group.get_local_rank()]

    monkeypatch.setattr(native_cp.spmd, "shard", local_shard)
    sharded = model._cp_shard(
        {"positions": positions, "attention_masks": metadata},
        input_shardings={"positions": decoder_input_sharding()["positions"]},
        parallelism_context=SimpleNamespace(get_mesh=lambda axis: mesh),
        parallelism=config.parallelism,
    )
    prepared = sharded["attention_masks"]
    assert isinstance(prepared, metadata_owner) and prepared is not metadata
    assert prepared.q.seqused.tolist() == [4]
    assert sharded["positions"].tolist() == [4, 5, 6, 7]
    with pytest.raises(RuntimeError, match=r"only Decoder\._cp_shard publishes"):
        cp_shard_context()
