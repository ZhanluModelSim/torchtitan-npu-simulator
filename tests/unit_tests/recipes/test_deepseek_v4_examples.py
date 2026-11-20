# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve example recipes through the production configuration loader."""

import pytest
from torchtitan.config.loader import ConfigLoader

import torchtitan_npu  # noqa: F401

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def example_environment(monkeypatch):
    for name in (
        "CLI_OVERRIDES",
        "OPTIMIZER_OVERRIDES",
        "WORLD_SIZE",
        "NGPU",
        "NNODES",
        "NODE_IPS",
        "CP",
        "TP",
        "PP",
        "EP",
        "DP_SHARD",
        "HF_ASSETS_PATH",
        "DATA_FILES",
        "DATASET_PATH",
        "DSV4_SFT_DATASET",
        "DSV4_PER_DOC_ALIGNMENT",
        "USE_GOLDEN",
        "STEPS",
        "CKPT_SAVE_LOAD_PATH",
        "CKPT_INIT_LOAD_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def _load(name, *, recipe_module=None):
    if recipe_module is not None:
        return ConfigLoader().load(["--module", recipe_module, "--config", name])
    if name.removesuffix("_reference") in {
        "deepseek_v4_debugmodel",
        "deepseek_v4_flash_43layers_16experts_eager_bf16",
        "deepseek_v4_flash_43layers_16experts_eager_fp8",
        "deepseek_v4_flash_1p_2k",
    }:
        return ConfigLoader().load(["--module", "torchtitan_npu_recipes.tests.models.deepseek_v4", "--config", name])
    return ConfigLoader().load(["--module", "torchtitan_npu_recipes.models.deepseek_v4", "--config", name])


def test_reduced_dsv4_recipe_surface_contains_only_verified_entries():
    from torchtitan_npu_recipes.tests.models import deepseek_v4

    assert {
        name for name, value in vars(deepseek_v4).items() if name.startswith("deepseek_v4_") and callable(value)
    } == {
        "deepseek_v4_debugmodel",
        "deepseek_v4_flash_43layers_16experts_eager_bf16",
        "deepseek_v4_flash_43layers_16experts_eager_fp8",
        "deepseek_v4_flash_1p_2k",
    }


def test_full_a5_recipe_keeps_explicit_resources_and_prequantization_contract(monkeypatch):
    monkeypatch.setenv("CKPT_INIT_LOAD_PATH", "/hf-base")
    monkeypatch.setenv("CKPT_SAVE_LOAD_PATH", "/native-checkpoints")
    config = _load("deepseek_v4_flash_cpt_4k_a5")

    assert config.parallelism.expert_parallel_degree == 128
    assert config.parallelism.data_parallel_shard_degree == 128
    assert config.training.num_tokens_per_train_step == 4194304
    assert config.checkpointer.load_only
    assert config.checkpointer.folder == "/path/to/save_ckpt"
    assert config.checkpointer.initial_load_path == "/path/to/init_load_ckpt"
    assert config.checkpointer.initial_load_in_hf
    assert config.extension.quantization.fsdp_prequantize_fqns == [
        ".attention.wq_b",
        ".attention.wo_a",
        ".attention.indexer.wq_b",
        ".feed_forward.routed_experts.w13",
        ".feed_forward.routed_experts.w2",
    ]
    kv = config.extension.quantization.kv_norm_quantization
    assert kv.format == "mxfp8" and kv.block_size == 64
    assert kv.fqns == [
        ".attention.kv_norm",
        ".attention.compressor.norm",
        ".attention.compressor_128.norm",
    ]


def test_qat_recipe_preserves_inherited_a5_policy_and_adds_mxfp4():
    config = _load("deepseek_v4_flash_qat_4k_a5")

    assert config.extension.quantization.enable_mxfp4_qat
    assert config.extension.quantization.li_quantization == "mxfp4"
    assert config.extension.quantization.enable_fsdp_prequantize
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 8192


@pytest.mark.parametrize(
    "name,message,recipe_module",
    [
        ("deepseek_v4_flash_cpt_1024k_a5", "Context parallelism is deferred", None),
    ],
    ids=["context-parallel"],
)
def test_deferred_examples_report_the_actual_unavailable_capability(name, message, recipe_module):
    with pytest.raises(NotImplementedError, match=message):
        _load(name, recipe_module=recipe_module)


@pytest.mark.parametrize(
    "name",
    [
        "deepseek_v4_flash_cpt_4k_a3",
        "deepseek_v4_flash_cpt_4k_a5",
        "deepseek_v4_flash_qat_4k_a5",
        "deepseek_v4_flash_sft_4k_a3",
        "deepseek_v4_flash_lora_4k_a3",
    ],
)
def test_full_model_recipes_load_master_training_defaults(name):
    from torchtitan.distributed.activation_remat import DefaultActivationRematConfig, FullActivationRematConfig

    config = _load(name)
    assert config.model.n_layers == 43
    assert config.training.max_context_length == 4096
    assert config.training.num_tokens_per_train_step == 1024 * 4096
    assert config.training.steps == 100
    assert config.parallelism.expert_parallel_degree == 128
    assert config.parallelism.data_parallel_shard_degree == 128
    assert config.parallelism.data_parallel_replicate_degree == 1
    from torchtitan.components.optim import AdamW

    from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon

    optimizer = config.optim.optimizer
    muon = next(group for group in optimizer.optimizers if isinstance(group, DistMuon.Config))
    adamw = next(group for group in optimizer.optimizers if isinstance(group, AdamW.Config))
    assert (muon.lr, muon.momentum, muon.ns_steps) == (1e-5, 0.95, 10)
    assert (adamw.lr, adamw.betas, adamw.eps) == (1e-5, (0.9, 0.95), 1e-8)
    assert adamw.weight_decay == 0.1
    assert config.compile.model.backend == "inductor"
    assert config.comm.init_timeout_seconds == 7200
    assert config.comm.train_timeout_seconds == 600
    assert config.override.imports[0] == (
        "torchtitan_npu.override.deepseek_v4.common.rms_norm"
        if name.endswith("a5")
        else "torchtitan_npu.override.common.rms_norm.asc"
    )
    assert config.override.imports[1:5] == [
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {"selector_backend": "asc"}),
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
    ]
    if name.endswith("a5"):
        from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher

        for layer in config.model.layers:
            dispatcher = layer.feed_forward.routed_experts.token_dispatcher
            assert isinstance(dispatcher, DeepEPTokenDispatcher.Config)
            assert dispatcher.num_max_tokens_per_rank == 8192
            assert dispatcher.hidden_dim == 4096
            assert dispatcher.absorb_router_scores
        assert isinstance(config.activation_remat, DefaultActivationRematConfig)
        assert config.override.imports[-2:] == [
            "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
            "torchtitan_npu.override.common.token_dispatcher.asc_deepep",
        ]
        assert not any("swap_optimizer" in item for item in config.override.imports)
    else:
        from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

        assert isinstance(
            config.model.layers[0].feed_forward.routed_experts.token_dispatcher, AllToAllTokenDispatcher.Config
        )
        assert isinstance(config.activation_remat, FullActivationRematConfig)
        assert config.override.imports[5:] == [
            "torchtitan_npu.override.common.optimizer.swap_optimizer",
            "torchtitan_npu.override.common.rope.asc_complex",
            "torchtitan_npu.override.common.token_dispatcher.asc",
        ]
    assert config.checkpointer.load_only == ("sft" not in name and "lora" not in name)


def test_full_recipes_allow_resource_edits_without_affecting_later_configs():
    first = _load("deepseek_v4_flash_cpt_4k_a5")
    first.hf_assets_path = "/tokenizer"
    first.dataloader.dataset.dataset.source.load_dataset_kwargs["data_files"] = "/data.jsonl"
    first.checkpointer.folder = "/checkpoints"
    first.checkpointer.initial_load_path = "/weights"
    first.override.imports.clear()
    first.extension.quantization.fsdp_prequantize_fqns.clear()

    second = _load("deepseek_v4_flash_cpt_4k_a5")
    assert second.hf_assets_path == "/path/to/DeepSeekV4_tokenizer"
    assert (
        second.dataloader.dataset.dataset.source.load_dataset_kwargs["data_files"] == "tests/assets/c4_test/data.json"
    )
    assert second.checkpointer.folder == "/path/to/save_ckpt"
    assert second.checkpointer.initial_load_path == "/path/to/init_load_ckpt"
    assert second.override.imports
    assert second.extension.quantization.fsdp_prequantize_fqns
