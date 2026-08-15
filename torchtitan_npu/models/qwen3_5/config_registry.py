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
from torchtitan.models.qwen3_5.config_registry import qwen35_27b
from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5
from torchtitan.trainer import Trainer

_TEXT_SPECIAL_TOKEN_IDS = {"image_id": -1, "video_id": -1}


def _messages(sample):
    return sample["messages"]


def _text_inputs(_model, args, kwargs):
    kwargs.setdefault("special_tokens", _TEXT_SPECIAL_TOKEN_IDS)
    return args, kwargs


def _parallelize_text(model, **kwargs):
    model.register_forward_pre_hook(_text_inputs, with_kwargs=True)
    return parallelize_qwen3_5(model, **kwargs)


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
