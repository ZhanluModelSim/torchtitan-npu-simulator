# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve example recipes through the production configuration loader."""

import pytest
from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.config.loader import ConfigLoader
from torchtitan.distributed.activation_remat import FullActivationRematConfig

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


def _load(name):
    return ConfigLoader().load(["--module", "torchtitan_npu_recipes.examples.deepseek_v4", "--config", name])


def test_eager_example_resolves_fixture_assets_without_a_cli_override():
    config = _load("deepseek_v4_eager_1p")

    assert config.hf_assets_path == "tests/assets/deepseek_v3"
    assert config.training.max_context_length == 2048
    assert config.compile is None
    assert config.checkpointer is None


@pytest.mark.parametrize("golden", [False, True], ids=["npu", "reference"])
def test_a3_flash_eager_preserves_upstream_training_policy(monkeypatch, golden):
    monkeypatch.setenv("HF_ASSETS_PATH", "/model-assets")
    monkeypatch.setenv("USE_GOLDEN", str(int(golden)))
    config = _load("deepseek_v4_flash_8p_cpt_4k_a3_eager")

    assert config.compile is None
    assert config.model.local_compile_regions == []
    assert isinstance(config.activation_remat, FullActivationRematConfig)
    assert config.activation_remat.save_regions == []
    assert config.training.max_context_length == config.model.max_context_length == 4096
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 4096
    assert config.training.num_tokens_per_train_step == 262144
    assert config.training.steps == 100
    assert config.parallelism.expert_parallel_degree == 8
    assert config.parallelism.data_parallel_shard_degree == 8
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.dataloader.max_num_documents is None
    muon, adamw = config.optim.optimizer.optimizers
    assert isinstance(muon, DistMuon.Config)
    assert muon.lr == 1e-5 and muon.ns_steps == 10
    assert isinstance(adamw, AdamW.Config)
    assert adamw.betas == (0.9, 0.95) and adamw.eps == 1e-8


@pytest.mark.parametrize("golden", [False, True], ids=["npu-muon", "reference-adamw"])
def test_mini_example_selects_actual_optimizer_and_operator_policy(monkeypatch, golden):
    monkeypatch.setenv("USE_GOLDEN", str(int(golden)))
    monkeypatch.setenv("HF_ASSETS_PATH", "/model-assets")
    config = _load("deepseek_v4_mini_1p_cpt_2k_a3")

    assert config.hf_assets_path == "/model-assets"
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 2048
    assert config.training.num_tokens_per_train_step == -1
    assert config.training.steps == 100
    assert config.parallelism.expert_parallel_degree == 1
    assert config.dataloader.dataset.dataset.processor.per_doc_alignment == 128
    if golden:
        assert isinstance(config.optim.optimizer.optimizers[0], AdamW.Config)
        assert config.optim.optimizer.optimizers[0].lr == 1e-5
        assert config.override.imports == ["torchtitan_npu.override.common.rope.workaround"]
    else:
        muon, adamw = config.optim.optimizer.optimizers
        assert isinstance(muon, DistMuon.Config)
        assert muon.lr == 1e-5 and muon.ns_steps == 10
        assert isinstance(adamw, AdamW.Config)
        assert adamw.betas == (0.9, 0.95) and adamw.eps == 1e-8
        assert config.override.imports[-1] == "torchtitan_npu.override.common.optimizer.swap_optimizer"
        assert (
            "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
            {"num_heads": 16, "index_num_heads": 8, "selector_backend": "asc"},
        ) in config.override.imports


@pytest.mark.parametrize(
    "name,microbatch,quantization,swap",
    [
        ("deepseek_v4_flash_8p_cpt_4k_a3", 4096, None, True),
        ("deepseek_v4_flash_8p_cpt_4k_a5", 8192, "all_block_fp8", False),
        ("deepseek_v4_flash_8p_cpt_4k_a5_hif8", 4096, "all_hif8", True),
    ],
    ids=["a3", "a5-block-fp8", "a5-hif8"],
)
def test_flash_examples_resolve_training_parallelism_and_quantization(name, microbatch, quantization, swap):
    config = _load(name)

    assert config.training.num_tokens_per_microbatch_per_dp_rank == microbatch
    assert config.training.num_tokens_per_train_step == 262144
    assert config.parallelism.expert_parallel_degree == 8
    assert config.parallelism.data_parallel_shard_degree == 8
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.compile.model.backend == "inductor"
    assert ("torchtitan_npu.override.common.optimizer.swap_optimizer" in config.override.imports) is swap
    assert config.extension.quantization.enable_quantized_training is (quantization is not None)
    if quantization:
        assert config.extension.quantization.recipe == quantization
        assert config.activation_remat.save_regions == ["*"]
        assert config.activation_remat.recompute_regions == ["*routed_experts.w13.*"]


def test_full_a5_recipe_keeps_checkpoint_and_prequantization_contract(monkeypatch):
    monkeypatch.setenv("CKPT_INIT_LOAD_PATH", "/hf-base")
    monkeypatch.setenv("CKPT_SAVE_LOAD_PATH", "/native-checkpoints")
    config = _load("deepseek_v4_flash_cpt_4k_a5")

    assert config.parallelism.expert_parallel_degree == 128
    assert config.parallelism.data_parallel_shard_degree == 128
    assert config.training.num_tokens_per_train_step == 4194304
    assert config.checkpointer.load_only
    assert config.checkpointer.folder == "/native-checkpoints"
    assert config.checkpointer.initial_load_path == "/hf-base"
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
    assert kv.fqns == [".attention.kv_norm", ".attention.compressor.norm"]


def test_qat_recipe_preserves_inherited_a5_policy_and_adds_mxfp4():
    config = _load("deepseek_v4_flash_qat_4k_a5")

    assert config.extension.quantization.enable_mxfp4_qat
    assert config.extension.quantization.li_quantization == "mxfp4"
    assert config.extension.quantization.enable_fsdp_prequantize
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 8192


def test_sft_example_keeps_its_conversation_loader_and_writable_checkpoints(monkeypatch):
    monkeypatch.setenv("DATASET_PATH", "/conversations.jsonl")
    config = _load("deepseek_v4_flash_8p_sft_4k_a3")

    assert config.dataloader.dataset.dataset.source.load_dataset_kwargs["data_files"] == "/conversations.jsonl"
    assert config.dataloader.dataset.dataset.processor.per_doc_alignment == 128
    assert config.training.num_tokens_per_train_step == 262144
    assert config.checkpointer is not None and not config.checkpointer.load_only


def test_pro_offload_example_retains_distinct_ep_and_fsdp_degrees():
    config = _load("deepseek_v4_pro_8p_cpt_4k_a5_cpuoffload")

    assert config.parallelism.expert_parallel_degree == 4
    assert config.parallelism.data_parallel_shard_degree == 8
    assert config.training.enable_cpu_offload
    assert config.optim.optimizer._cpu_offload
    assert config.training.num_tokens_per_train_step == 262144


def test_legacy_override_environment_fails_with_a_migration_action(monkeypatch):
    monkeypatch.setenv("CLI_OVERRIDES", "my_module.my_override")

    with pytest.raises(ValueError, match=r"Python recipe.*--override"):
        _load("deepseek_v4_mini_1p_cpt_2k_a3")


@pytest.mark.parametrize(
    "name,message",
    [
        ("deepseek_v4_flash_cpt_1024k_a5", "Context parallelism is deferred"),
        ("deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer", "NPU GraphTrainer remains deferred"),
        ("deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel", "no longer exposes"),
    ],
    ids=["context-parallel", "graph-config-layout", "cv-backend"],
)
def test_deferred_examples_report_the_actual_unavailable_capability(name, message):
    with pytest.raises(NotImplementedError, match=message):
        _load(name)


def test_compiled_example_preserves_the_current_training_guard():
    config = _load("deepseek_v4_flash_8p_cpt_4k_a3")

    with pytest.raises(NotImplementedError, match="NPU model compilation is not supported"):
        config.__post_init__()
