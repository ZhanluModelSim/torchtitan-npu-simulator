# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve V4.1 example policies through the real Python configuration loader."""

from pathlib import Path

import pytest
from torchtitan.components.optim import DistMuon
from torchtitan.config.loader import ConfigLoader

import torchtitan_npu  # noqa: F401
from torchtitan_npu.models.deepseek_v4_1.optimizer import EngramTableOptimizerConfig

_MODULE = "torchtitan_npu_recipes.models.deepseek_v4_1"
_GRAPH_MODULE = "torchtitan_npu_recipes.tests.graph_trainer.deepseek_v4_1"
_DEBUG_MODULE = "torchtitan_npu_recipes.tests.models.deepseek_v4_1"


@pytest.fixture
def example_environment(monkeypatch):
    for name in (
        "WORLD_SIZE",
        "NGPU",
        "NNODES",
        "NODE_IPS",
        "EP",
        "CP",
        "DP_SHARD",
        "STEPS",
        "USE_GOLDEN",
        "USE_TILELANG",
        "CLI_OVERRIDES",
        "OPTIMIZER_OVERRIDES",
        "DATALOADER_OVERRIDES",
        "MHC_POST_OVERRIDE",
        "DATA_FILES",
        "DATASET_PATH",
        "HF_ASSETS_PATH",
        "DSV41_PER_DOC_ALIGNMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_ASSETS_PATH", str(Path(__file__).resolve().parents[3] / "tests/assets/deepseek_v3"))
    return monkeypatch


@pytest.mark.cpu
def test_mini_recipe_resolves_training_budget_and_quantized_policy(example_environment, tmp_path):
    example_environment.setenv("STEPS", "7")
    assets = tmp_path / "assets with spaces"
    assets.mkdir()
    example_environment.setenv("HF_ASSETS_PATH", str(assets))

    config = ConfigLoader().load(["--module", _DEBUG_MODULE, "--config", "deepseek_v4_1_mini_1p_cpt_a5"])

    assert config.hf_assets_path == str(assets)
    assert config.model.max_context_length == config.training.max_context_length == 2048
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 2048
    assert config.training.num_tokens_per_train_step == 2048
    assert config.training.steps == 7
    assert config.parallelism.data_parallel_shard_degree == 1
    assert config.parallelism.expert_parallel_degree == 1
    assert config.extension.quantization.enable_quantized_training
    assert config.extension.quantization.recipe == "all_block_fp8"
    assert config.compile is None
    assert "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc" in config.override.imports
    assert "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc" in config.override.imports


@pytest.mark.cpu
@pytest.mark.parametrize("custom_topology", [False, True], ids=["released-cp128", "two-rank-cp2"])
def test_long_context_recipe_rejects_cp_until_its_lifecycle_is_migrated(example_environment, custom_topology):
    if custom_topology:
        example_environment.setenv("NGPU", "2")
        example_environment.setenv("NNODES", "1")
        example_environment.setenv("EP", "2")
        example_environment.setenv("CP", "2")

    with pytest.raises(NotImplementedError, match=r"context parallelism requires the v0\.4 CP lifecycle migration"):
        ConfigLoader().load(["--module", _MODULE, "--config", "deepseek_v4_1_flash_cpt_1m_a5"])


@pytest.mark.cpu
def test_long_context_recipe_preserves_text_input_and_optimizer_at_cp1(example_environment, tmp_path):
    data = tmp_path / "packed text.jsonl"
    data.write_text('{"text": "An independent text document."}\n', encoding="utf-8")
    example_environment.setenv("DATASET_PATH", str(data))
    for name in ("NGPU", "NNODES", "EP", "CP"):
        example_environment.setenv(name, "1")

    config = ConfigLoader().load(["--module", _MODULE, "--config", "deepseek_v4_1_flash_cpt_1m_a5"])

    assert config.model.max_context_length == config.training.max_context_length == 1048576
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 1048576
    assert config.training.num_tokens_per_train_step == 1048576
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 1
    assert config.parallelism.data_parallel_shard_degree == 1
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.dataloader.dataset.dataset.source.load_dataset_kwargs["data_files"] == str(data)
    assert config.dataloader.dataset.dataset.processor.per_doc_alignment == 256
    assert not config.engram_enabled
    assert all(layer.engram is None for layer in config.model.layers)
    assert isinstance(config.optim.optimizer.optimizers[0], DistMuon.Config)
    assert config.optim.optimizer.optimizers[0].ns_steps == 10
    assert not any(isinstance(group, EngramTableOptimizerConfig) for group in config.optim.optimizer.optimizers)
    assert config.compile is None
    # Layout declarations are complete before Trainer applies the requested overrides.
    assert config.model.layers[0].feed_forward.routed_experts.w13.sharding_config is not None


@pytest.mark.cpu
def test_legacy_override_text_requires_an_explicit_recipe(example_environment):
    example_environment.setenv("CLI_OVERRIDES", "torchtitan_npu.override.common.rope.workaround")

    with pytest.raises(ValueError, match=r"CLI_OVERRIDES.*Python recipe"):
        ConfigLoader().load(["--module", _DEBUG_MODULE, "--config", "deepseek_v4_1_mini_1p_cpt_a5"])


@pytest.mark.cpu
@pytest.mark.parametrize(
    "a5,multimodal",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["a3-text", "a3-multimodal", "a5-text", "a5-multimodal"],
)
def test_graph_recipe_preserves_model_owner_selects_pipeline_and_applies_overrides(example_environment, a5, multimodal):
    from torchtitan.config import apply_overrides

    from torchtitan_npu.models.deepseek_v4_1.model import (
        GraphTrainerDeepSeekV41Model,
        GraphTrainerDeepSeekV41MultimodalModel,
    )
    from torchtitan_npu.models.deepseek_v4_1.trainer import DeepSeekV41GraphTrainer
    from torchtitan_npu.override.common.rope import AscComplexRoPE, AscPartialComplexRoPE
    from torchtitan_npu.override.deepseek_v4_1.vision import AscUnmaskedVisionSDPA

    # CPU recipe validation exercises factories and conflict detection, not
    # the CANN JIT loader; device execution belongs to NPU smoke/ST.
    example_environment.setattr("torchtitan_npu.override.common.swiglu_group._ensure_cann_ops_loaded", lambda: None)

    name = f"graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_{'a5' if a5 else 'a3'}" + ("_multimodal" if multimodal else "")
    config = ConfigLoader().load(["--module", _GRAPH_MODULE, "--config", name])

    assert type(config) is DeepSeekV41GraphTrainer.Config
    model_cls = GraphTrainerDeepSeekV41MultimodalModel if multimodal else GraphTrainerDeepSeekV41Model
    assert type(config.model) is model_cls.Config
    assert config.engram_enabled
    assert [i for i, layer in enumerate(config.model.layers) if layer.engram is not None] == [1, 14]
    assert any(isinstance(group, EngramTableOptimizerConfig) for group in config.optim.optimizer.optimizers)
    assert config.training.num_tokens_per_train_step == -1
    assert config.training.num_tokens_per_microbatch_per_dp_rank == config.model.max_context_length == 4096
    assert config.compile.ep_overlap.enabled
    assert config.compile.ep_overlap.module_fqn == "layers.*.feed_forward"
    assert config.compile.ep_overlap.chunk_dim == "seq"
    assert config.compile.pass_pipeline == "mutation-functionalization+npu_auto_overlap"
    from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY

    assert callable(PASS_PIPELINE_REGISTRY["mutation-functionalization"])
    assert callable(PASS_PIPELINE_REGISTRY[config.compile.pass_pipeline])
    assert ("torchtitan_npu.override.deepseek_v4_1.vision.asc_unmasked_sdpa" in config.override.imports) == multimodal
    assert config.model.layers[0].feed_forward.router.num_experts == (24 if a5 else 16)
    assert config.extension.quantization.enable_quantized_training == a5
    if a5:
        assert config.extension.quantization.recipe == "all_block_fp8"
        assert "torchtitan_npu.override.common.swiglu_group.asc" in config.override.imports
        assert "torchtitan_npu.override.common.swiglu_group.asc_shared_experts" in config.override.imports
        assert "torchtitan_npu.override.common.rope.asc_partial" in config.override.imports

    # Trainer applies the complete list to the root, so matching different
    # implementation names must not leave conflicting claims on the same node.
    apply_overrides(config.override, config)
    rope_cls = AscPartialComplexRoPE if a5 else AscComplexRoPE
    assert type(config.model.layers[0].attention.rope) is rope_cls.Config
    if multimodal:
        assert type(config.model.vision_encoder.unmasked_sdpa) is AscUnmaskedVisionSDPA.Config
    if a5:
        from torchtitan_npu.override.common.swiglu_group import AscSwiGLU

        assert type(config.model.layers[0].feed_forward.routed_experts.activation_fn) is AscSwiGLU.Config
        assert type(config.model.layers[0].feed_forward.shared_experts.activation_fn) is AscSwiGLU.Config


@pytest.mark.cpu
@pytest.mark.parametrize("source", ["a3", "a5", "public"], ids=["a3", "a5", "public"])
def test_multimodal_graph_recipe_applies_vision_leaf_override(example_environment, source):
    from torchtitan.config import OverrideConfig
    from torchtitan.config.override import apply_overrides

    from torchtitan_npu.override.deepseek_v4_1.vision import AscUnmaskedVisionSDPA

    name = f"graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_{source}_multimodal"
    module = _GRAPH_MODULE
    if source == "public":
        name = "graph_trainer_deepseek_v4_1_flash_40layers_16experts_multimodal"
        module = "torchtitan_npu_recipes.graph_trainer.deepseek_v4_1"
    config = ConfigLoader().load(["--module", module, "--config", name])
    target = "torchtitan_npu.override.deepseek_v4_1.vision.asc_unmasked_sdpa"
    entries = [
        entry
        for entry in config.override.imports
        if (entry[0] if isinstance(entry, tuple) else entry) == target
    ]
    assert len(entries) == 1
    apply_overrides(OverrideConfig(imports=entries), config.model.vision_encoder)
    assert type(config.model.vision_encoder.unmasked_sdpa) is AscUnmaskedVisionSDPA.Config


def test_graph_recipe_preserves_base_extension_configuration(example_environment, monkeypatch):
    from torchtitan_npu_recipes.graph_trainer import deepseek_v4_1 as recipes

    factory = recipes._make_trainer_config
    base = None

    def configured_base(*args, **kwargs):
        nonlocal base
        base = factory(*args, **kwargs)
        base.extension.quantization.enable_quantized_training = True
        base.extension.quantization.recipe = "all_block_fp8"
        return base

    monkeypatch.setattr(recipes, "_make_trainer_config", configured_base)
    config = recipes.graph_trainer_deepseek_v4_1_flash_40layers_16experts_text()
    assert config.extension is base.extension
    assert config.extension.quantization.enable_quantized_training
    assert config.extension.quantization.recipe == "all_block_fp8"
