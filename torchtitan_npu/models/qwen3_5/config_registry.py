# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os
from dataclasses import replace

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.config_registry import qwen35_27b
from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5
from torchtitan.trainer import Trainer

from torchtitan_npu.override.qwen3_5.parallelize import parallelize_qwen3_5_cp

_TEXT_SPECIAL_TOKEN_IDS = {"image_id": -1, "video_id": -1}


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
    return parallelize_qwen3_5_cp(model, **kwargs)


def qwen35_27b_4k_sft() -> Trainer.Config:
    config = qwen35_27b()
    assert config.model_spec is not None
    config.model_spec = replace(config.model_spec, parallelize_fn=_parallelize_text)
    config.dataloader = ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": os.environ["DATA_FILES"], "split": "train"},
        sample_processor=_messages,
        infinite=True,
    )
    config.optimizer = default_adamw(lr=1e-4, weight_decay=0.0)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=16, decay_ratio=1.0, min_lr_factor=0.1)
    config.training.local_batch_size = 1
    config.training.seq_len = 4096
    config.training.steps = 256
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpoint.interval = 256
    config.checkpoint.last_save_model_only = False
    config.metrics.log_freq = 1
    return config


def qwen35_27b_long_text_sft() -> Trainer.Config:
    config = qwen35_27b_4k_sft()
    config.model_spec = replace(model_registry("27B", attn_backend="varlen"), parallelize_fn=_parallelize_long_text)
    config.dataloader = ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": os.environ["DATA_FILES"].split(","), "split": "train"},
        sample_processor=_long_text_messages,
        infinite=True,
    )
    config.optimizer = default_adamw(lr=1e-4, weight_decay=0.1)
    config.training.max_norm = 1.0
    config.training.seq_len = 32768
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
