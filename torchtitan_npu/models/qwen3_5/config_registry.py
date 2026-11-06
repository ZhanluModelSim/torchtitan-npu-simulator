# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import copy
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.config import derive
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS, MultiModalProcessor
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.models.qwen3_5 import config_registry as _upstream_config_registry
from torchtitan.models.qwen3_5.config_registry import qwen35_27b
from torchtitan.trainer import Trainer

from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager

from . import NPUQwen35Model, model_registry

if TYPE_CHECKING:
    from renderers import Message

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
    dataset_path = os.environ.get("QWEN_DATASET_PATH") or os.environ.get("DATASET_PATH")
    if dataset_path is not None and not dataset_path.strip():
        raise ValueError("QWEN_DATASET_PATH / DATASET_PATH must be non-empty when set")
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


def qwen35_debugmodel(seq_len: int | None = _upstream_config_registry.DEFAULT_DEBUG_MODEL_SEQ_LEN) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel(seq_len=seq_len)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_debugmodel_moe(
    seq_len: int | None = _upstream_config_registry.DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_122b_a10b(seq_len: int | None = None) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_122b_a10b(seq_len=seq_len)
    # The released model has two KV heads; TP must divide that count.
    config.parallelism.tensor_parallel_degree = 2
    return config


def qwen35_397b_a17b(seq_len: int | None = None) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_397b_a17b(seq_len=seq_len)
    config.parallelism.tensor_parallel_degree = 2
    return config


def qwen35_debugmodel_moe_float8_lora(seq_len: int | None = 512) -> Trainer.Config:
    """LoRA on NPU block-FP8 weights with block-32 activation quantization."""
    from torchtitan.config.transform import (
        GroupedLinearLoRAHandler,
        LinearLoRAHandler,
        LoRATransform,
        apply_transforms,
    )
    from torchtitan.models.common.linear import GroupedLinear, Linear

    from torchtitan_npu.extensions.adaptor.torchao.param_swaps import block_fp8_param_swap
    from torchtitan_npu.extensions.adaptor.torchao.recipes import NpuQuantizeConverter

    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.model = model_registry(
        "debugmodel_moe",
        enable_sp=True,
        seq_len=seq_len,
        moe_comm_backend="standard",
        converters=[
            NpuQuantizeConverter.Config(
                base_config=block_fp8_param_swap(),
                unified_npu_quant_modules=(Linear.Config, GroupedLinear.Config),
                model_compile_enabled=False,
            )
        ],
    )
    return apply_transforms(
        config,
        [
            LoRATransform(
                handlers=(LinearLoRAHandler(), GroupedLinearLoRAHandler()),
                rank=8,
                alpha=16.0,
                target_modules=["w13", "w2"],
            )
        ],
    )


def _messages(sample: dict[str, Any]) -> list["Message"]:
    return cast("list[Message]", sample["messages"])


def _long_text_messages(sample: dict[str, Any]) -> list["Message"]:
    if "messages" in sample:
        return cast("list[Message]", sample["messages"])
    if "prompt" in sample:
        return cast(
            "list[Message]",
            [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": sample["response"]}],
        )
    user = sample["instruction"]
    if sample.get("input"):
        user = f"{user}\n\n{sample['input']}"
    return cast(
        "list[Message]",
        [{"role": "user", "content": user}, {"role": "assistant", "content": sample["output"]}],
    )


def qwen35_27b_4k_sft() -> Trainer.Config:
    config = qwen35_27b(seq_len=4096)
    assert config.model is not None
    config.model = derive(config.model, NPUQwen35Model.Config, text_only=True)
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={"data_files": os.environ.get("DATA_FILES", "tests/assets/c4_test/data.json")},
                ),
                processor=ChatProcessor.Config(messages_fn=_messages),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=1,
        ),
    )
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-4, weight_decay=0.0)])
    config.optim.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=16, decay_ratio=1.0, min_lr_factor=0.1)
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.max_context_length = 4096
    config.training.steps = 256
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpointer = CheckpointManager.Config(interval=256, last_save_model_only=False)
    config.metrics.log_freq = 1
    return config


def qwen35_27b_long_text_sft() -> Trainer.Config:
    config = qwen35_27b_4k_sft()
    config.model = derive(
        model_registry("27B", enable_sp=False, seq_len=32768, attn_backend="varlen"),
        NPUQwen35Model.Config,
        text_only=True,
        vision_encoder=None,
    )
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={
                        "data_files": os.environ.get("DATA_FILES", "tests/assets/c4_test/data.json").split(",")
                    },
                ),
                processor=ChatProcessor.Config(messages_fn=_long_text_messages),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=1,
        ),
    )
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-4, weight_decay=0.1)])
    config.optim.max_norm = 1.0
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
    from torchtitan.config.transform import ContextParallelTransform
    from torchtitan.models.common.cp_attention import UlyssesCPVarlenInnerAttention

    transformed = ContextParallelTransform(inner_attention=UlyssesCPVarlenInnerAttention).transform(config.model)
    assert isinstance(transformed, NPUQwen35Model.Config)
    config.model = transformed
    config.checkpointer = replace(
        config.checkpointer, interval=1000, last_save_model_only=True, last_save_in_hf=True, export_dtype="bfloat16"
    )
    return config
