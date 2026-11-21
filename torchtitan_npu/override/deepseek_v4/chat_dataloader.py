# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import replace
from glob import escape
from pathlib import Path

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.config import override

from torchtitan_npu.models.common.alignment import AlignedTextProcessor
from torchtitan_npu.models.common.sft_shard import SFTMapDatasetConfig
from torchtitan_npu.models.deepseek_v4.chat_encoder import DSV4EncoderConfig, ReasoningEffort, ThinkingMode
from torchtitan_npu.models.deepseek_v4.chat_processor import DeepSeekV4ChatProcessor


@override(target=GrainDataLoader.Config, fqns=["dataloader"], exact=True, description="DeepSeek V4 chat SFT")
def sft(
    cfg: GrainDataLoader.Config,
    *,
    dataset_path: str | None = None,
    thinking_mode: ThinkingMode = "thinking",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
) -> GrainDataLoader.Config:
    packed = cfg.dataset
    if not isinstance(packed, ConcatThenSplitPackingConfig) or not isinstance(packed.dataset, SingleDatasetConfig):
        raise ValueError("DeepSeek V4 chat SFT requires the CPT text data recipe")
    samples = packed.dataset
    if not isinstance(samples.processor, AlignedTextProcessor.Config):
        raise ValueError("DeepSeek V4 chat SFT requires the CPT text processor")

    dataset_path = dataset_path or os.environ.get("DATASET_PATH")
    path = Path(dataset_path).resolve() if dataset_path else None
    suffix = path.suffix.lower() if path else ""
    if path is None or suffix not in (".jsonl", ".json"):
        raise ValueError("Set DATASET_PATH to a DeepSeek V4 chat JSONL or JSON file")
    pattern = escape(str(path))
    source = (
        IndexedJsonlSource.Config(patterns=(pattern,))
        if suffix == ".jsonl"
        else HuggingFaceRandomAccessSource.Config(
            path="json", split="train", load_dataset_kwargs={"data_files": pattern}
        )
    )
    processor = DeepSeekV4ChatProcessor.Config(
        per_doc_alignment=samples.processor.per_doc_alignment,
        chat_encoder=DSV4EncoderConfig(
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_default_bos_token,
            reasoning_effort=reasoning_effort,
        ),
    )
    dataset = SFTMapDatasetConfig(
        source=source, pre_filters=samples.pre_filters, processor=processor, post_filters=samples.post_filters
    )
    return replace(cfg, dataset=FirstFitPackingConfig(dataset=dataset))
