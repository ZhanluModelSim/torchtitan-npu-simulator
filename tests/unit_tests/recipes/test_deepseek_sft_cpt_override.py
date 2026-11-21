# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import json
from dataclasses import fields
from glob import escape
from importlib import import_module

import pytest
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides, parse_cli_imports
from torchtitan.trainer import Trainer

from torchtitan_npu.models.deepseek_v4.chat_processor import DeepSeekV4ChatProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import _GrainSFTProcessor

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("hardware", ["a3", "a5"])
@pytest.mark.parametrize("vision", [False, True], ids=["v4", "v4_1"])
@pytest.mark.parametrize("indexed", [True, False], ids=["jsonl", "json-or-parquet"])
def test_sft_overrides_only_cpt_dataloader(tmp_path, monkeypatch, hardware, vision, indexed):
    suffix = ".jsonl" if indexed else (".parquet" if vision else ".json")
    data = tmp_path / f"chat [train]{suffix}"
    monkeypatch.setenv("DATASET_PATH", str(data))
    monkeypatch.setenv("WORLD_SIZE", "128")
    monkeypatch.setenv("CP", "1")
    if vision:
        monkeypatch.setenv("DATASET", "cc12m-test")
        monkeypatch.setenv("PACKING_BUFFER_SIZE", "2")
        monkeypatch.setenv("DP_SHARD", "128")
        monkeypatch.setenv("EP", "128")
    model = "deepseek_v4_1" if vision else "deepseek_v4"
    name = f"{model}_flash_cpt_4k_{hardware}"
    module = import_module(f"torchtitan_npu_recipes.models.{model}")
    loader_name = "vision_language_dataloader" if vision else "chat_dataloader"
    target = f"torchtitan_npu.override.{model}.{loader_name}.sft"

    config = getattr(module, name)()
    assert isinstance(config, Trainer.Config)
    before = config.to_dict()
    cpt_loader = config.dataloader
    options = {"thinking_mode": "thinking" if vision else "chat", "reasoning_effort": "high"}
    imports = parse_cli_imports([f"{target}={json.dumps(options)}"])
    replacements = apply_overrides(OverrideConfig(imports=imports), config)
    assert len(replacements) == 1
    after = config.to_dict()
    before.pop("dataloader")
    after.pop("dataloader")
    assert after == before
    loader = config.dataloader
    assert isinstance(loader, GrainDataLoader.Config)
    assert all(
        getattr(loader, field.name) == getattr(cpt_loader, field.name)
        for field in fields(loader)
        if field.name not in {"dataset", "collator"}
    )
    dataset = loader.dataset
    if not vision:
        assert isinstance(dataset, FirstFitPackingConfig)
        dataset = dataset.dataset
    assert isinstance(dataset, SingleDatasetConfig)
    processor = dataset.processor
    if vision:
        assert isinstance(processor, _GrainSFTProcessor.Config)
        encoder = processor
    else:
        assert isinstance(processor, DeepSeekV4ChatProcessor.Config)
        encoder = processor.chat_encoder
    assert encoder.thinking_mode == options["thinking_mode"]
    assert encoder.reasoning_effort == options["reasoning_effort"]
    pattern = escape(str(data.resolve()))
    if indexed:
        assert isinstance(dataset.source, IndexedJsonlSource.Config)
        assert dataset.source.patterns == (pattern,)
    elif vision:
        assert isinstance(dataset.source, HuggingFaceStreamingSource.Config)
        assert dataset.source.load_dataset_kwargs["data_files"] == {"train": pattern}
    else:
        assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
        assert dataset.source.load_dataset_kwargs["data_files"] == pattern
