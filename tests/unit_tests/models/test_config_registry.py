# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torchtitan", reason="upstream torchtitan is required")


@dataclass(frozen=True, slots=True)
class ModelConfigRegistry:
    module_name: str
    config_model_names: tuple[str, ...]
    enabled: bool
    example_dir: str | None = None
    example_model_names: tuple[str, ...] = ()


CONFIG_REGISTRIES = (
    ModelConfigRegistry(
        module_name="torchtitan_npu.models.deepseek_v4.config_registry",
        config_model_names=("deepseek_v4_flash", "deepseek_v4_pro", "deepseek_v4"),
        example_dir="examples/deepseek_v4",
        example_model_names=("deepseek_v4_flash", "deepseek_v4_pro", "deepseek_v4"),
        enabled=True,
    ),
    # Reserved for the same generic checks once DeepSeek V3 registry / examples are migrated.
    ModelConfigRegistry(
        module_name="torchtitan_npu.models.deepseek_v3.config_registry",
        config_model_names=("deepseek_v3_671b", "deepseek_v3"),
        enabled=False,
    ),
)

ENABLED_CONFIG_REGISTRIES = tuple(registry for registry in CONFIG_REGISTRIES if registry.enabled)

# Naming rules follow RFC: https://gitcode.com/cann/torchtitan-npu/issues/73


def _model_name_pattern(model_names: tuple[str, ...]) -> str:
    return "(?:" + "|".join(re.escape(model_name) for model_name in model_names) + ")"


def _config_registry_module(registry: ModelConfigRegistry) -> ModuleType:
    return importlib.import_module(registry.module_name)


def _config_prefixes(registry: ModelConfigRegistry) -> tuple[str, ...]:
    return tuple(
        prefix
        for model_name in registry.config_model_names
        for prefix in (model_name, f"sft_{model_name}", f"debug_{model_name}")
    )


def _debug_config_prefixes(registry: ModelConfigRegistry) -> tuple[str, ...]:
    return tuple(f"debug_{model_name}" for model_name in registry.config_model_names)


def _hf_weight_load_prefixes(registry: ModelConfigRegistry) -> tuple[str, ...]:
    return tuple(
        prefix
        for model_name in registry.config_model_names
        for prefix in (model_name, f"sft_{model_name}")
    )


def _public_config_names(registry: ModelConfigRegistry, prefixes: tuple[str, ...]) -> list[str]:
    config_registry = _config_registry_module(registry)
    return sorted(
        name
        for name in dir(config_registry)
        if name.startswith(prefixes) and callable(getattr(config_registry, name))
    )


def _config_name_pattern(registry: ModelConfigRegistry) -> re.Pattern[str]:
    model_name = _model_name_pattern(registry.config_model_names)
    seq_len = r"[0-9]+k"
    cluster_size = r"[0-9]+die"
    suffix = r"[a-z0-9]+"
    return re.compile(
        rf"^(?:"
        rf"{model_name}_{seq_len}_{cluster_size}(?:_{suffix})?"
        rf"|sft_{model_name}_{seq_len}_{cluster_size}_{suffix}"
        rf"|debug_{model_name}_(?:single_node(?:_{suffix})?|smoketest)"
        rf")$"
    )


def _example_name_pattern(registry: ModelConfigRegistry) -> re.Pattern[str]:
    model_name = _model_name_pattern(registry.example_model_names)
    seq_len = r"[0-9]+k"
    platform = r"[A-Za-z0-9]+"
    suffix = r"[a-z0-9]+"
    return re.compile(
        rf"^(?:"
        rf"cpt_{model_name}_{seq_len}_{platform}"
        rf"|sft_{model_name}_{seq_len}_{platform}"
        rf"|debug_{model_name}_single_node(?:_{suffix})?"
        rf")\.sh$"
    )


def _example_script_names(registry: ModelConfigRegistry) -> list[str]:
    assert registry.example_dir is not None
    repo_root = Path(__file__).resolve().parents[3]
    example_dir = repo_root / registry.example_dir
    task_prefixes = ("cpt_", "sft_", "debug_")
    return sorted(path.name for path in example_dir.glob("*.sh") if path.name.startswith(task_prefixes))


def _example_shell_scripts() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return sorted((repo_root / "examples").rglob("*.sh"))


def _hyphenated_cli_tokens(script_path: Path) -> list[str]:
    option_pattern = re.compile(r"(?<![\w-])--[A-Za-z0-9][A-Za-z0-9_.-]*")
    subcommand_pattern = re.compile(r"(?<![\w-])[a-z][A-Za-z0-9_.]*:[a-z][A-Za-z0-9_-]*")
    script_text = script_path.read_text()
    cli_tokens = {*option_pattern.findall(script_text), *subcommand_pattern.findall(script_text)}
    return sorted(token for token in cli_tokens if "-" in token.removeprefix("--"))


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_config_registry_names_follow_rfc_patterns(registry: ModelConfigRegistry):
    config_names = _public_config_names(registry, _config_prefixes(registry))
    pattern = _config_name_pattern(registry)

    assert config_names
    for config_name in config_names:
        assert pattern.fullmatch(config_name), config_name


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_config_registry_names_reject_legacy_snode_suffix(registry: ModelConfigRegistry):
    pattern = _config_name_pattern(registry)

    assert not pattern.fullmatch(f"debug_{registry.config_model_names[0]}_snode")


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_example_script_names_follow_rfc_patterns(registry: ModelConfigRegistry):
    if registry.example_dir is None:
        pytest.skip(f"{registry.module_name} has no example script naming rules yet")

    example_names = _example_script_names(registry)
    pattern = _example_name_pattern(registry)

    assert example_names
    for example_name in example_names:
        assert pattern.fullmatch(example_name), example_name


def test_example_scripts_use_underscore_cli_options():
    bad_options = {}
    for script_path in _example_shell_scripts():
        hyphenated_cli_tokens = _hyphenated_cli_tokens(script_path)
        if hyphenated_cli_tokens:
            bad_options[str(script_path)] = hyphenated_cli_tokens

    assert bad_options == {}


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_public_config_functions_return_trainer_config(registry: ModelConfigRegistry):
    from torchtitan_npu.config.configs import TrainerConfig

    config_registry = _config_registry_module(registry)
    config_names = _public_config_names(registry, _config_prefixes(registry))

    assert config_names
    for config_name in config_names:
        assert isinstance(getattr(config_registry, config_name)(), TrainerConfig), config_name


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_debug_configs_do_not_default_to_hf_weight_loading(registry: ModelConfigRegistry):
    config_registry = _config_registry_module(registry)
    config_names = _public_config_names(registry, _debug_config_prefixes(registry))

    assert config_names
    for config_name in config_names:
        trainer_config = getattr(config_registry, config_name)()

        assert trainer_config.checkpoint.enable is False
        assert trainer_config.checkpoint.initial_load_in_hf is False
        assert trainer_config.checkpoint.initial_load_path is None


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_train_configs_default_to_hf_weight_loading(registry: ModelConfigRegistry):
    config_registry = _config_registry_module(registry)
    config_names = _public_config_names(registry, _hf_weight_load_prefixes(registry))

    assert config_names
    for config_name in config_names:
        trainer_config = getattr(config_registry, config_name)()

        assert trainer_config.checkpoint.enable is True
        assert trainer_config.checkpoint.initial_load_in_hf is True


def _deepseek_v4_registry() -> ModuleType:
    return importlib.import_module("torchtitan_npu.models.deepseek_v4.config_registry")


def _converter_names(trainer_config) -> list[str]:
    return [getattr(converter, "name", type(converter).__name__) for converter in trainer_config.model_converters.converters]


def test_trainer_base_config_sets_shared_training_defaults():
    from torchtitan_npu.config import configs
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

    trainer_config = configs.trainer_base_config()

    assert isinstance(trainer_config.dataloader, HuggingFaceTextDataLoader.Config)
    assert trainer_config.dataloader.dataset == "c4_test"
    assert trainer_config.debug.print_config is True
    assert trainer_config.metrics.log_freq == 1
    assert trainer_config.optimizer.lr == 1e-5
    assert trainer_config.optimizer.eps == 1e-6
    assert trainer_config.optimizer.swap_optimizer is True
    assert trainer_config.lr_scheduler.warmup_steps == 200
    assert trainer_config.lr_scheduler.decay_ratio == 0.8
    assert trainer_config.lr_scheduler.decay_type == "cosine"
    assert trainer_config.lr_scheduler.min_lr_factor == 0.01
    assert trainer_config.training.local_batch_size == 1
    assert trainer_config.training.global_batch_size == -1
    assert trainer_config.training.seq_len == 4096
    assert trainer_config.training.steps == 10000
    assert trainer_config.training.num_mtp_modules == 0
    assert trainer_config.parallelism.fsdp_reshard_after_forward == "always"
    assert trainer_config.activation_checkpoint.mode == "full"


def test_deepseek_v4_flash_cpt_config_applies_flash_and_cpt_overrides():
    config_registry = _deepseek_v4_registry()
    trainer_config = config_registry.deepseek_v4_flash_4k_128die()

    assert trainer_config.model_spec.name == "deepseek_v4"
    assert trainer_config.model_spec.flavor == "v4_flash_debug_256_experts_43_layers"
    assert _converter_names(trainer_config) == [
        "npu_rms_norm",
        "npu_moe_dispatch",
        "npu_gmm",
        "npu_rope",
        "npu_smla",
        "npu_mhc_pre",
    ]
    assert trainer_config.comm.trace_buf_size == 0
    assert trainer_config.lr_scheduler.warmup_steps == 400
    assert trainer_config.training.seq_len == 4096
    assert trainer_config.training.steps == 2000
    assert trainer_config.training.global_batch_size == 1024
    assert trainer_config.training.num_mtp_modules == 1
    assert trainer_config.parallelism.data_parallel_shard_degree == 128
    assert trainer_config.parallelism.expert_parallel_degree == 64
    assert trainer_config.checkpoint.enable is True
    assert trainer_config.checkpoint.initial_load_in_hf is True
    assert trainer_config.checkpoint.load_only is True
    assert trainer_config.checkpoint.interval == 10000


def test_deepseek_v4_pro_cpt_config_only_overrides_pro_differences():
    config_registry = _deepseek_v4_registry()
    flash_config = config_registry.deepseek_v4_flash_4k_128die()
    pro_config = config_registry.deepseek_v4_pro_4k_384die()

    assert pro_config.model_spec.name == flash_config.model_spec.name
    assert pro_config.model_spec.flavor == "v4_pro_debug_61_layers"
    assert pro_config.model_converters == flash_config.model_converters
    assert pro_config.comm == flash_config.comm
    assert pro_config.lr_scheduler == flash_config.lr_scheduler
    assert pro_config.checkpoint == flash_config.checkpoint
    assert pro_config.training.steps == flash_config.training.steps
    assert pro_config.training.seq_len == flash_config.training.seq_len
    assert pro_config.training.num_mtp_modules == flash_config.training.num_mtp_modules
    assert pro_config.training.global_batch_size == 384
    assert pro_config.parallelism.data_parallel_shard_degree == 384
    assert pro_config.parallelism.expert_parallel_degree == 192
    assert pro_config.compile.enable is True


def test_debug_single_node_config_applies_common_debug_semantics_without_cpt_comm_override():
    config_registry = _deepseek_v4_registry()
    trainer_config = config_registry.debug_deepseek_v4_flash_single_node()

    assert trainer_config.model_spec.flavor == "v4_flash_debug_16_experts_43_layers"
    assert _converter_names(trainer_config) == [
        "npu_rms_norm",
        "npu_moe_dispatch",
        "npu_gmm",
        "npu_rope",
        "npu_smla",
        "npu_mhc_pre",
        "npu_mhc_post",
    ]
    assert trainer_config.debug.print_config is True
    assert trainer_config.debug.moe_force_load_balance is True
    assert trainer_config.comm.trace_buf_size != 0
    assert trainer_config.lr_scheduler.warmup_steps == 4
    assert trainer_config.training.seq_len == 4096
    assert trainer_config.training.steps == 20
    assert trainer_config.training.global_batch_size == -1
    assert trainer_config.training.num_mtp_modules == 1
    assert trainer_config.parallelism.expert_parallel_degree == 8
    assert trainer_config.checkpoint.enable is False
    assert trainer_config.checkpoint.initial_load_in_hf is False
    assert trainer_config.checkpoint.load_only is True
    assert trainer_config.compile.enable is False


def test_deepseek_v4_sft_processors_follow_upstream_hf_dataset_layout():
    processors = importlib.import_module("torchtitan_npu.hf_datasets.chat_processors")
    config_registry = _deepseek_v4_registry()
    tau_config = config_registry.sft_deepseek_v4_flash_16k_128die_tau()
    gsm8k_config = config_registry.sft_deepseek_v4_flash_1k_128die_gsm8k()

    assert callable(processors.process_tau_sample)
    assert callable(processors.process_gsm8k_sample)
    assert tau_config.dataloader.sample_processor is processors.process_tau_sample
    assert gsm8k_config.dataloader.sample_processor is processors.process_gsm8k_sample


def test_tau_sample_processor_accepts_json_messages_and_tools():
    from torchtitan_npu.hf_datasets.chat_processors import process_tau_sample

    messages = [{"role": "system", "content": "existing"}, {"role": "user", "content": "book a flight"}]
    tools = [{"name": "search_flights", "description": "Search flights"}]

    processed = process_tau_sample({"messages": json.dumps(messages), "tools": json.dumps(tools)})

    assert processed == [
        {"role": "system", "content": "existing", "tools": tools},
        {"role": "user", "content": "book a flight"},
    ]
    assert messages[0] == {"role": "system", "content": "existing"}


def test_tau_sample_processor_inserts_system_message_for_tools():
    from torchtitan_npu.hf_datasets.chat_processors import process_tau_sample

    tools = [{"name": "lookup", "description": "Lookup data"}]

    processed = process_tau_sample({"messages": [{"role": "user", "content": "lookup order"}], "tools": tools})

    assert processed == [
        {"role": "system", "content": "", "tools": tools},
        {"role": "user", "content": "lookup order"},
    ]


def test_gsm8k_sample_processor_splits_reasoning_and_final_answer():
    from torchtitan_npu.hf_datasets.chat_processors import process_gsm8k_sample

    processed = process_gsm8k_sample({"question": "What is 2+2?", "answer": "Compute 2 + 2. #### 4"})

    assert processed == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Compute 2 + 2.", "content": "4"},
    ]


def test_deepseek_v4_debug_configs_do_not_inherit_cpt_comm_override():
    config_registry = _deepseek_v4_registry()

    for config_name in ("debug_deepseek_v4_flash_single_node", "debug_deepseek_v4_pro_single_node"):
        trainer_config = getattr(config_registry, config_name)()

        assert trainer_config.comm.trace_buf_size != 0


def test_deepseek_v4_flash_single_node_mxfp8_derives_from_flash_single_node_config():
    from torchtitan.components.quantization.mx import MXFP8Converter

    config_registry = _deepseek_v4_registry()

    flash_config = config_registry.debug_deepseek_v4_flash_single_node()
    mxfp8_config = config_registry.debug_deepseek_v4_flash_single_node_mxfp8()

    assert mxfp8_config.model_spec == flash_config.model_spec
    assert mxfp8_config.parallelism == flash_config.parallelism
    assert mxfp8_config.training == flash_config.training
    assert mxfp8_config.checkpoint == flash_config.checkpoint
    assert _converter_names(mxfp8_config)[:-1] == _converter_names(flash_config)
    assert len(mxfp8_config.model_converters.converters) == len(flash_config.model_converters.converters) + 1
    assert isinstance(mxfp8_config.model_converters.converters[-1], MXFP8Converter.Config)
    assert mxfp8_config.model_converters.converters[-1].recipe_name == "mxfp8_rceil"


def test_deepseek_v4_pro_single_node_derives_from_flash_single_node_config():
    config_registry = _deepseek_v4_registry()

    flash_config = config_registry.debug_deepseek_v4_flash_single_node()
    pro_config = config_registry.debug_deepseek_v4_pro_single_node()

    assert pro_config.model_spec.name == flash_config.model_spec.name
    assert pro_config.model_spec.flavor == "v4_pro_debug_16_layers"
    assert pro_config.training == flash_config.training
    assert pro_config.checkpoint == flash_config.checkpoint
    assert pro_config.debug == flash_config.debug
    assert pro_config.comm == flash_config.comm
    assert _converter_names(pro_config) == _converter_names(flash_config)[:-1]
    assert pro_config.parallelism.expert_parallel_degree == 16


def test_deepseek_v4_sft_configs_keep_flash_cpt_semantics():
    config_registry = _deepseek_v4_registry()
    flash_cpt = config_registry.deepseek_v4_flash_4k_128die()

    for config_name in ("sft_deepseek_v4_flash_16k_128die_tau", "sft_deepseek_v4_flash_1k_128die_gsm8k"):
        trainer_config = getattr(config_registry, config_name)()

        assert trainer_config.model_spec.name == flash_cpt.model_spec.name
        assert trainer_config.model_spec.flavor == flash_cpt.model_spec.flavor
        assert trainer_config.comm.trace_buf_size == 0
        assert trainer_config.debug.moe_force_load_balance is False
        assert trainer_config.lr_scheduler.warmup_steps == 20
        assert trainer_config.training.steps == 100
        assert trainer_config.training.global_batch_size == -1
        assert trainer_config.training.num_mtp_modules == flash_cpt.training.num_mtp_modules
        assert trainer_config.parallelism.data_parallel_shard_degree == flash_cpt.parallelism.data_parallel_shard_degree
        assert trainer_config.parallelism.pipeline_parallel_schedule == "1F1B"
        assert trainer_config.checkpoint.initial_load_in_hf is True
        assert trainer_config.checkpoint.load_only is False
        assert trainer_config.checkpoint.export_dtype == "bfloat16"


def test_deepseek_v4_sft_configs_apply_dataset_specific_overrides():
    from torchtitan_npu.config.configs import ChatDataLoaderConfig

    config_registry = _deepseek_v4_registry()
    tau_config = config_registry.sft_deepseek_v4_flash_16k_128die_tau()
    gsm8k_config = config_registry.sft_deepseek_v4_flash_1k_128die_gsm8k()

    assert isinstance(tau_config.dataloader, ChatDataLoaderConfig)
    assert tau_config.training.seq_len == 16384
    assert tau_config.parallelism.context_parallel_degree == 4
    assert tau_config.dataloader.load_dataset_kwargs == {"data_files": "train-00000-of-00001.parquet", "split": "train"}

    assert isinstance(gsm8k_config.dataloader, ChatDataLoaderConfig)
    assert gsm8k_config.training.seq_len == 1024
    assert gsm8k_config.parallelism.context_parallel_degree == 1
    assert gsm8k_config.dataloader.load_dataset_kwargs == {"name": "main", "split": "train"}
