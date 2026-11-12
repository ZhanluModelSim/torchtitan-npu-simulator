# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen example configuration selection through the native recipe loader."""

import pytest
from torchtitan.components.data import HuggingFaceStreamingSource
from torchtitan.config.loader import ConfigLoader

import torchtitan_npu  # noqa: F401
from torchtitan_npu.models.qwen3_5 import NPUQwen35Model


@pytest.fixture
def qwen_resources(monkeypatch, tmp_path):
    for name in (
        "DATASET",
        "DATASET_PATH",
        "QWEN_DATASET_PATH",
        "HF_ASSETS_PATH",
        "OVERRIDE_IMPORTS",
        "ENABLE_NPU_MOE_DISPATCHER",
        "WORLD_SIZE",
        "NGPU",
    ):
        monkeypatch.delenv(name, raising=False)
    upstream = tmp_path / "upstream"
    (upstream / "tests/assets/tokenizer").mkdir(parents=True)
    monkeypatch.setenv("TORCHTITAN_REPO", str(upstream))
    return upstream


@pytest.mark.cpu
@pytest.mark.parametrize(
    "recipe,microbatch,step_tokens,ep,dp",
    [
        ("qwen3_5_debugmodel_1p_cpt_512", 512, 512, 1, -1),
        ("qwen3_5_debugmodel_moe_8p_cpt_512", 1024, -1, 8, 8),
    ],
    ids=["dense-1p", "moe-8p"],
)
def test_loader_selects_qwen_example_training_and_resources(recipe, microbatch, step_tokens, ep, dp, qwen_resources):
    config = ConfigLoader().load(["--module", "torchtitan_npu_recipes.examples.qwen3_5", "--config", recipe])

    assert isinstance(config.model, NPUQwen35Model.Config)
    assert config.model.max_context_length == config.training.max_context_length == 512
    assert config.training.num_tokens_per_microbatch_per_dp_rank == microbatch
    assert config.training.num_tokens_per_train_step == step_tokens
    assert config.training.steps == 10
    assert config.training.disable_cuda_graphs
    assert config.debug.print_config
    assert config.optim.optimizer.optimizers[0].lr == 5e-3
    assert config.parallelism.expert_parallel_degree == ep
    assert config.parallelism.data_parallel_shard_degree == dp
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.context_parallel_degree == 1
    assert config.hf_assets_path == str(qwen_resources / "tests/assets/tokenizer")
    assert isinstance(config.dataloader.dataset.source, HuggingFaceStreamingSource.Config)
    assert config.dataloader.dataset.source.path == str(qwen_resources / "tests/assets/cc12m_test")
    assert config.override.imports == ["torchtitan_npu.override.qwen3_5.gated_delta.npu"]


@pytest.mark.cpu
@pytest.mark.parametrize("size_variable", ["NGPU", "WORLD_SIZE"])
def test_moe_recipe_replicates_complete_eight_rank_groups(size_variable, monkeypatch, qwen_resources):
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv(size_variable, "16")
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_npu_recipes.examples.qwen3_5",
            "--config",
            "qwen3_5_debugmodel_moe_8p_cpt_512",
        ]
    )

    assert config.parallelism.data_parallel_replicate_degree == 2
    assert config.parallelism.data_parallel_shard_degree == 8
    assert config.parallelism.expert_parallel_degree == 8


@pytest.mark.cpu
def test_moe_recipe_rejects_incomplete_eight_rank_groups(monkeypatch, qwen_resources):
    monkeypatch.setenv("WORLD_SIZE", "12")
    with pytest.raises(ValueError, match="positive multiple of 8"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_npu_recipes.examples.qwen3_5",
                "--config",
                "qwen3_5_debugmodel_moe_8p_cpt_512",
            ]
        )


@pytest.mark.cpu
def test_loader_honors_qwen_asset_paths_and_dispatcher_selection(monkeypatch, tmp_path, qwen_resources):
    assets = tmp_path / "model-assets"
    assets.mkdir()
    monkeypatch.setenv("HF_ASSETS_PATH", str(assets))
    monkeypatch.setenv("DATASET_PATH", str(tmp_path / "samples"))
    monkeypatch.setenv("ENABLE_NPU_MOE_DISPATCHER", "1")

    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_npu_recipes.examples.qwen3_5",
            "--config",
            "qwen3_5_debugmodel_moe_8p_cpt_512",
        ]
    )

    assert config.hf_assets_path == str(assets)
    assert config.dataloader.dataset.source.path == str(tmp_path / "samples")
    assert config.override.imports == [
        "torchtitan_npu.override.qwen3_5.gated_delta.npu",
        "torchtitan_npu.override.common.token_dispatcher.asc",
    ]


@pytest.mark.cpu
def test_legacy_qwen_override_environment_requires_explicit_configuration(monkeypatch, qwen_resources):
    monkeypatch.setenv("OVERRIDE_IMPORTS", "")
    with pytest.raises(ValueError, match=r"config\.override\.imports.*--override"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_npu_recipes.examples.qwen3_5",
                "--config",
                "qwen3_5_debugmodel_1p_cpt_512",
            ]
        )


@pytest.mark.cpu
def test_qwen36_selector_keeps_27b_migration_requirement():
    with pytest.raises(NotImplementedError, match=r"Qwen3\.5 27B recipe requires migration"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_npu_recipes.examples.qwen3_6",
                "--config",
                "run_qwen3_6_4k",
            ]
        )
