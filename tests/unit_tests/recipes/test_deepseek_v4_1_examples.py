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

_MODULE = "torchtitan_npu_recipes.examples.deepseek_v4_1"


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

    config = ConfigLoader().load(["--module", _MODULE, "--config", "deepseek_v4_1_mini_1p_cpt_a5"])

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
def test_long_context_recipe_preserves_text_input_and_explicit_cp_policy(
    example_environment, tmp_path, custom_topology
):
    data = tmp_path / "packed text.jsonl"
    data.write_text('{"text": "An independent text document."}\n', encoding="utf-8")
    example_environment.setenv("DATASET_PATH", str(data))
    if custom_topology:
        example_environment.setenv("NGPU", "2")
        example_environment.setenv("NNODES", "1")
        example_environment.setenv("EP", "2")
        example_environment.setenv("CP", "2")
    expected_degree = 2 if custom_topology else 128

    config = ConfigLoader().load(["--module", _MODULE, "--config", "deepseek_v4_1_flash_cpt_1m_a5"])

    assert config.model.max_context_length == config.training.max_context_length == 1048576
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 1048576
    assert config.training.num_tokens_per_train_step == 1048576
    assert config.parallelism.context_parallel_degree == expected_degree
    assert config.parallelism.expert_parallel_degree == expected_degree
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
    # The fused stack supports context parallelism, so the recipe configures it.
    config.model.update_from_config(config=config)


@pytest.mark.cpu
def test_legacy_override_text_requires_an_explicit_recipe(example_environment):
    example_environment.setenv("CLI_OVERRIDES", "torchtitan_npu.override.common.rope.workaround")

    with pytest.raises(ValueError, match=r"CLI_OVERRIDES.*Python recipe"):
        ConfigLoader().load(["--module", _MODULE, "--config", "deepseek_v4_1_mini_1p_cpt_a5"])
