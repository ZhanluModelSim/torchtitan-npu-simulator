# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import copy
import os
from dataclasses import replace

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS, MultiModalProcessor
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.models.qwen3_5 import config_registry as _upstream_config_registry
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.config_registry import qwen35_27b
from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5
from torchtitan.trainer import Trainer

from torchtitan_npu.override.qwen3_5.parallelize import parallelize_qwen3_5_cp

_TEXT_SPECIAL_TOKEN_IDS = {"image_id": -1, "video_id": -1}


# Keep upstream benchmark recipes available through the NPU module.
# The debug factories below configure the native dataset from launcher inputs;
# NPU kernels remain explicitly selected through override.imports.
for _name, _factory in vars(_upstream_config_registry).items():
    if _name.startswith("qwen35_") and callable(_factory) and _name not in globals():
        globals()[_name] = _factory


def _multimodal_dataloader() -> GrainDataLoader.Config:
    """Build the multimodal loader for DATASET, honouring DATASET_PATH.

    The dataset, collator and shuffle policy are the upstream multimodal ones;
    this constructs the loader directly instead of patching a text loader.
    """
    dataset_name = os.environ.get("DATASET", "cc12m-test")
    if dataset_name not in MM_DATASETS:
        raise ValueError(f"Unknown DATASET {dataset_name!r}; expected one of {sorted(MM_DATASETS)}")
    dataset = copy.deepcopy(MM_DATASETS[dataset_name])
    dataset_path = os.environ.get("DATASET_PATH")
    if dataset_path:
        source = dataset.source
        assert isinstance(source, HuggingFaceStreamingSource.Config)
        source.path = dataset_path
    processor = dataset.processor
    assert isinstance(processor, MultiModalProcessor.Config)
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(
            build_mrope_positions=True,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
        ),
        streaming_shuffle_buffer_size=128,
    )


def qwen35_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel(seq_len=seq_len)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_debugmodel_moe(seq_len: int | None = None) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.dataloader = _multimodal_dataloader()
    return config


def _messages(sample):
    return sample["messages"]


def _long_text_messages(sample):
    if "messages" in sample:
        return sample["messages"]
    if "prompt" in sample:
        return [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": sample["response"]}]
    user = sample["instruction"]
    if sample.get("input"):
        user = f"{user}\n\n{sample['input']}"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": sample["output"]}]


def _text_inputs(_model, args, kwargs):
    kwargs.setdefault("special_tokens", _TEXT_SPECIAL_TOKEN_IDS)
    return args, kwargs


def _parallelize_text(model, **kwargs):
    model.register_forward_pre_hook(_text_inputs, with_kwargs=True)
    return parallelize_qwen3_5(model, **kwargs)


def _parallelize_long_text(model, **kwargs):
    model.register_forward_pre_hook(_text_inputs, with_kwargs=True)
    # Long-text SFT is text-only: detach the VL vision encoder so the CP
    # sequence-metadata adapter only ever sees pure-text varlen masks.
    model.vision_encoder = None
    return parallelize_qwen3_5_cp(model, **kwargs)


def qwen35_27b_4k_sft() -> Trainer.Config:
    config = qwen35_27b(seq_len=4096)
    assert config.model_spec is not None
    config.model_spec = replace(config.model_spec, parallelize_fn=_parallelize_text)
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={"data_files": os.environ["DATA_FILES"]},
                ),
                processor=ChatProcessor.Config(messages_fn=_messages),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=1,
        ),
    )
    config.optimizer = default_adamw(lr=1e-4, weight_decay=0.0)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=16, decay_ratio=1.0, min_lr_factor=0.1)
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.max_context_length = 4096
    config.training.steps = 256
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpoint.interval = 256
    config.checkpoint.last_save_model_only = False
    config.metrics.log_freq = 1
    return config


def qwen35_27b_long_text_sft() -> Trainer.Config:
    config = qwen35_27b_4k_sft()
    config.model_spec = replace(
        model_registry("27B", seq_len=32768, attn_backend="varlen"), parallelize_fn=_parallelize_long_text
    )
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={"data_files": os.environ["DATA_FILES"].split(",")},
                ),
                processor=ChatProcessor.Config(messages_fn=_long_text_messages),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=1,
        ),
    )
    config.optimizer = default_adamw(lr=1e-4, weight_decay=0.1)
    config.training.max_norm = 1.0
    config.training.num_tokens_per_microbatch_per_dp_rank = 32768
    config.training.max_context_length = 32768
    config.parallelism.context_parallel_degree = 4
    config.parallelism.context_parallel_load_balancer = None
    config.override.imports.extend(
        [
            "torchtitan_npu.override.qwen3_5.varlen_attention.asc_cp",
            "torchtitan_npu.override.qwen3_5.gated_delta.context_parallel",
        ]
    )
    config.checkpoint = replace(
        config.checkpoint, interval=1000, last_save_model_only=True, last_save_in_hf=True, export_dtype="bfloat16"
    )
    return config
