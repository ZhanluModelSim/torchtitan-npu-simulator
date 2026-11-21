# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import replace
from glob import escape
from pathlib import Path
from typing import cast

from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.mm_datasets import MMSamplePackingConfig
from torchtitan.config import override

from torchtitan_npu.models.common.sft_shard import SFTMapDatasetConfig
from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import CaptionProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import sft_dataloader_config
from torchtitan_npu.models.deepseek_v4_1.vision.language_encoder import ReasoningEffort, ThinkingMode


@override(
    target=GrainDataLoader.Config, fqns=["dataloader"], exact=True, description="DeepSeek V4.1 vision-language SFT"
)
def sft(
    cfg: GrainDataLoader.Config,
    *,
    dataset_path: str | None = None,
    image_root: str = "",
    thinking_mode: ThinkingMode = "chat",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
) -> GrainDataLoader.Config:
    samples = cfg.dataset.dataset if isinstance(cfg.dataset, MMSamplePackingConfig) else cfg.dataset
    if not isinstance(samples, SingleDatasetConfig) or not isinstance(samples.processor, CaptionProcessor.Config):
        raise ValueError("DeepSeek V4.1 vision-language SFT requires the CPT multimodal data recipe")

    dataset_path = dataset_path or os.environ.get("DATASET_PATH")
    if not dataset_path:
        raise ValueError("Set DATASET_PATH to a DeepSeek V4.1 SFT JSONL, JSON, or Parquet file")
    spec = sft_dataloader_config(
        dataset_path=dataset_path,
        image_root=image_root,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
        per_doc_alignment=samples.processor.per_doc_alignment,
    )
    path = Path(dataset_path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return replace(cfg, dataset=spec.dataset, collator=spec.collator)
    pattern = escape(str(path))
    source = (
        IndexedJsonlSource.Config(patterns=(pattern,))
        if suffix == ".jsonl"
        else HuggingFaceRandomAccessSource.Config(
            path="json", split="train", load_dataset_kwargs={"data_files": pattern}
        )
    )
    stream = cast("SingleDatasetConfig", spec.dataset)
    dataset = SFTMapDatasetConfig(
        source=source, pre_filters=stream.pre_filters, processor=stream.processor, post_filters=stream.post_filters
    )
    return replace(cfg, dataset=dataset, collator=spec.collator)
