# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import re
import subprocess
import sys
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
    cluster_size = r"[0-9]+npus"
    suffix = r"[a-z0-9]+"
    return re.compile(
        rf"^(?:"
        rf"{model_name}_{seq_len}(?:_{suffix})*_{cluster_size}"
        rf"|sft_{model_name}_{seq_len}_{suffix}_{cluster_size}"
        rf"|debug_{model_name}_(?:single_node(?:_{suffix})?|smoketest)"
        rf"|debug_{model_name}(?:_{suffix})*_{cluster_size}"
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


def _qwen3_registry() -> ModuleType:
    return importlib.import_module("torchtitan_npu.models.qwen3.config_registry")


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
    trainer_config = config_registry.deepseek_v4_flash_4k_128npus()

    assert trainer_config.model_spec.name == "deepseek_v4"
    assert trainer_config.model_spec.flavor == "v4_flash_43layers_256experts"
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
    flash_config = config_registry.deepseek_v4_flash_4k_128npus()
    pro_config = config_registry.deepseek_v4_pro_4k_384npus()

    assert pro_config.model_spec.name == flash_config.model_spec.name
    assert pro_config.model_spec.flavor == "v4_pro_61layers_384experts"
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
    trainer_config = config_registry.debug_deepseek_v4_flash_8npus()

    assert trainer_config.model_spec.flavor == "v4_flash_43layers_16experts"
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


def test_deepseek_v4_chat_processors_are_registered():
    processors = importlib.import_module("torchtitan_npu.hf_datasets.chat_processors")

    assert callable(processors.process_tau_sample)
    assert callable(processors.process_gsm8k_sample)


def test_tau_demo_dataset_is_processable():
    pq = pytest.importorskip("pyarrow.parquet")
    from torchtitan_npu.hf_datasets.chat_processors import process_tau_sample

    table = pq.read_table("tests/assets/tau_historical_sft/demo_train_00000_of_00001.parquet")
    sample = table.slice(0, 1).to_pylist()[0]

    processed = process_tau_sample(sample)

    assert processed[0]["role"] == "system"
    assert processed[0]["tools"]
    assert any(message["role"] == "user" for message in processed)


def test_qwen3_sft_configs_use_registered_chat_processors():
    config_registry = _qwen3_registry()
    gsm8k_config = config_registry.sft_qwen3_30ba3b_gsm8k()
    wordle_config = config_registry.sft_qwen3_1_7b_wordle()

    assert gsm8k_config.dataloader.chat_processor == "torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample"
    assert wordle_config.dataloader.chat_processor == "torchtitan_npu.hf_datasets.chat_processors.process_wordle_sample"


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


def test_chat_processor_import_path_resolves_processors():
    processors = importlib.import_module("torchtitan_npu.hf_datasets.chat_processors")

    assert (
        processors.import_chat_processor("torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample")
        is processors.process_gsm8k_sample
    )
    assert (
        processors.import_chat_processor("torchtitan_npu.hf_datasets.chat_processors.process_tau_sample")
        is processors.process_tau_sample
    )
    assert (
        processors.import_chat_processor("torchtitan_npu.hf_datasets.chat_processors.process_wordle_sample")
        is processors.process_wordle_sample
    )

    with pytest.raises(ValueError, match="chat_processor must be an import path"):
        processors.import_chat_processor("missing")


def test_deepseek_v4_debug_configs_do_not_inherit_cpt_comm_override():
    config_registry = _deepseek_v4_registry()

    for config_name in ("debug_deepseek_v4_flash_8npus", "debug_deepseek_v4_pro_16npus"):
        trainer_config = getattr(config_registry, config_name)()

        assert trainer_config.comm.trace_buf_size != 0


def test_deepseek_v4_flash_single_node_mxfp8_derives_from_flash_single_node_config():
    from torchtitan.components.quantization.mx import MXFP8Converter

    config_registry = _deepseek_v4_registry()

    flash_config = config_registry.debug_deepseek_v4_flash_8npus()
    mxfp8_config = config_registry.debug_deepseek_v4_flash_mxfp8_8npus()

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

    flash_config = config_registry.debug_deepseek_v4_flash_8npus()
    pro_config = config_registry.debug_deepseek_v4_pro_16npus()

    assert pro_config.model_spec.name == flash_config.model_spec.name
    assert pro_config.model_spec.flavor == "v4_pro_16layers_16experts"
    assert pro_config.training == flash_config.training
    assert pro_config.checkpoint == flash_config.checkpoint
    assert pro_config.debug == flash_config.debug
    assert pro_config.comm == flash_config.comm
    assert _converter_names(pro_config) == _converter_names(flash_config)[:-1]
    assert pro_config.parallelism.expert_parallel_degree == 16


def test_deepseek_v4_cpt_config_can_select_chat_dataloader_from_cli():
    from torchtitan.config.manager import ConfigManager
    from torchtitan_npu.config.configs import ChatDataLoaderConfig

    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.models.deepseek_v4",
            "--config",
            "deepseek_v4_flash_4k_128npus",
            "--training.seq_len",
            "16384",
            "--parallelism.context_parallel_degree",
            "4",
            "--training.global_batch_size",
            "-1",
            "--checkpoint.no_load_only",
            "--checkpoint.interval",
            "500",
            "--checkpoint.export_dtype",
            "bfloat16",
            "dataloader:chat_data_loader_config",
            "--dataloader.dataset_path",
            "./tests/assets/tau_historical_sft",
            "--dataloader.data_files",
            "demo_train_00000_of_00001.parquet",
            "--dataloader.dataset_config_name",
            "default",
            "--dataloader.chat_processor",
            "torchtitan_npu.hf_datasets.chat_processors.process_tau_sample",
            "dataloader.chat_encoder:dsv4_encoder_config",
            "--dataloader.chat_encoder.encoding_module_path",
            "/tmp/encoding_dsv4.py",
        ]
    )

    assert isinstance(config.dataloader, ChatDataLoaderConfig)
    assert config.training.seq_len == 16384
    assert config.training.global_batch_size == -1
    assert config.parallelism.context_parallel_degree == 4
    assert config.checkpoint.load_only is False
    assert config.checkpoint.interval == 500
    assert config.checkpoint.export_dtype == "bfloat16"
    assert config.dataloader.dataset_path == "./tests/assets/tau_historical_sft"
    assert config.dataloader.data_files == "demo_train_00000_of_00001.parquet"
    assert config.dataloader.dataset_config_name == "default"
    assert config.dataloader.chat_processor == "torchtitan_npu.hf_datasets.chat_processors.process_tau_sample"
    assert config.dataloader.chat_encoder is not None
    assert config.dataloader.chat_encoder.encoding_module_path == "/tmp/encoding_dsv4.py"


def test_deepseek_v4_chat_dataloader_script_order_keeps_top_level_overrides_before_subcommands():
    from torchtitan.config.manager import ConfigManager

    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.models.deepseek_v4",
            "--config",
            "deepseek_v4_flash_4k_128npus",
            "--training.global_batch_size",
            "-1",
            "--checkpoint.no_load_only",
            "--checkpoint.interval",
            "500",
            "--checkpoint.export_dtype",
            "bfloat16",
            "--training.seq_len",
            "1024",
            "--parallelism.context_parallel_degree",
            "1",
            "dataloader:chat_data_loader_config",
            "--dataloader.dataset_path",
            "./tests/assets/tau_historical_sft",
            "--dataloader.data_files",
            "demo_train_00000_of_00001.parquet",
            "--dataloader.dataset_config_name",
            "default",
            "--dataloader.chat_processor",
            "torchtitan_npu.hf_datasets.chat_processors.process_tau_sample",
            "dataloader.chat_encoder:dsv4_encoder_config",
            "--dataloader.chat_encoder.encoding_module_path",
            "/tmp/encoding_dsv4.py",
        ]
    )

    assert config.training.global_batch_size == -1
    assert config.training.seq_len == 1024
    assert config.parallelism.context_parallel_degree == 1
    assert config.checkpoint.load_only is False
    assert config.checkpoint.interval == 500
    assert config.checkpoint.export_dtype == "bfloat16"
    assert config.dataloader.dataset_path == "./tests/assets/tau_historical_sft"
    assert config.dataloader.data_files == "demo_train_00000_of_00001.parquet"
    assert config.dataloader.dataset_config_name == "default"
    assert config.dataloader.chat_processor == "torchtitan_npu.hf_datasets.chat_processors.process_tau_sample"
    assert config.dataloader.chat_encoder.encoding_module_path == "/tmp/encoding_dsv4.py"


def test_deepseek_v4_cpt_config_can_select_gsm8k_chat_dataloader_from_cli():
    from torchtitan.config.manager import ConfigManager
    from torchtitan_npu.config.configs import ChatDataLoaderConfig

    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.models.deepseek_v4",
            "--config",
            "deepseek_v4_flash_4k_128npus",
            "--training.seq_len",
            "1024",
            "dataloader:chat_data_loader_config",
            "--dataloader.dataset_path",
            "/data/dataset/openai/gsm8k",
            "--dataloader.dataset_config_name",
            "main",
            "--dataloader.chat_processor",
            "torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample",
            "dataloader.chat_encoder:dsv4_encoder_config",
            "--dataloader.chat_encoder.encoding_module_path",
            "/tmp/encoding_dsv4.py",
        ]
    )

    assert isinstance(config.dataloader, ChatDataLoaderConfig)
    assert config.training.seq_len == 1024
    assert config.dataloader.dataset_path == "/data/dataset/openai/gsm8k"
    assert config.dataloader.dataset_config_name == "main"
    assert config.dataloader.chat_processor == "torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample"
    assert config.dataloader.chat_encoder is not None
    assert config.dataloader.chat_encoder.encoding_module_path == "/tmp/encoding_dsv4.py"


def test_chat_dataloader_help_exposes_one_typed_dataset_cli_surface():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchtitan.config.manager",
            "--module",
            "torchtitan_npu.models.deepseek_v4",
            "--config",
            "deepseek_v4_flash_4k_128npus",
            "dataloader:chat_data_loader_config",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    help_text = result.stdout + result.stderr
    assert "--dataloader.dataset-split" in help_text
    assert "--dataloader.data-files" in help_text
    assert "--dataloader.dataset-config-name" in help_text
    assert "--dataloader.load-dataset-kwargs" not in help_text
