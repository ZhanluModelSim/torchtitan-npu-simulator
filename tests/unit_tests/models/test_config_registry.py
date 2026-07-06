# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
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


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_config_registry_names_follow_rfc_patterns(registry: ModelConfigRegistry):
    config_names = _public_config_names(registry, _config_prefixes(registry))
    pattern = _config_name_pattern(registry)

    assert config_names
    for config_name in config_names:
        assert pattern.fullmatch(config_name), config_name


@pytest.mark.parametrize("registry", ENABLED_CONFIG_REGISTRIES, ids=lambda registry: registry.module_name)
def test_example_script_names_follow_rfc_patterns(registry: ModelConfigRegistry):
    if registry.example_dir is None:
        pytest.skip(f"{registry.module_name} has no example script naming rules yet")

    example_names = _example_script_names(registry)
    pattern = _example_name_pattern(registry)

    assert example_names
    for example_name in example_names:
        assert pattern.fullmatch(example_name), example_name


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
