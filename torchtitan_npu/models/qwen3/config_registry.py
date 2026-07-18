# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    ChatDataLoaderConfig,
    TrainerConfig,
    debug_single_node_eq_pruned_config,
    sft_default_config,
    trainer_base_config,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry

_DEFAULT_CONVERTERS = ("npu_rms_norm", "npu_rope", "npu_moe_dispatch", "npu_gmm")


def _default_converters(*extra_names: str) -> list:
    return [get_model_converter_config(name) for name in (*_DEFAULT_CONVERTERS, *extra_names)]


def _qwen3_06b_base() -> TrainerConfig:
    base = trainer_base_config()
    return replace(
        base,
        hf_assets_path="./tests/assets/tokenizer/qwen3-tokenizer",
        model_spec=model_registry("0.6B"),
        model_converters=ModelConvertersContainer.Config(converters=[]),
    )


def debug_qwen3_06b_single_node() -> TrainerConfig:
    base = debug_single_node_eq_pruned_config(_qwen3_06b_base())
    return replace(
        base,
        dataloader=replace(base.dataloader, dataset="c4_test"),
        checkpoint=replace(base.checkpoint, load_only=False),
    )


def _qwen3_30ba3b_base() -> TrainerConfig:
    base = trainer_base_config()
    return replace(
        base,
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        model_spec=model_registry("30B-A3B"),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
    )


def sft_qwen3_30ba3b_gsm8k() -> TrainerConfig:
    base = sft_default_config(_qwen3_30ba3b_base())
    return replace(
        base,
        optimizer=replace(base.optimizer, lr=1e-5),
        lr_scheduler=replace(base.lr_scheduler, warmup_steps=10, decay_ratio=0.9, min_lr_factor=0.1),
        training=replace(base.training, local_batch_size=1, seq_len=4096, steps=100),
        parallelism=replace(base.parallelism, expert_parallel_degree=8, context_parallel_degree=4),
        dataloader=ChatDataLoaderConfig(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            chat_processor="torchtitan_npu.hf_datasets.chat_processors.process_gsm8k_sample",
        ),
        checkpoint=replace(
            base.checkpoint,
            enable=True,
            initial_load_in_hf=True,
            initial_load_path="./assets/hf/Qwen3-30B-A3B",
        ),
        activation_checkpoint=replace(base.activation_checkpoint, mode="selective"),
    )


def sft_qwen3_30ba3b_gsm8k_tnd() -> TrainerConfig:
    """GSM8K + TND: same as math config but with NPUVarlenAttention."""
    from torchtitan_npu.models.qwen3.tnd_config import _enable_npu_varlen_attention

    base = sft_qwen3_30ba3b_gsm8k()
    return replace(
        base,
        model_spec=_enable_npu_varlen_attention(base.model_spec),
    )


def _qwen3_1_7b_converters() -> ModelConvertersContainer.Config:
    return ModelConvertersContainer.Config(converters=[])


def _qwen3_1_7b_base() -> TrainerConfig:
    base = trainer_base_config()
    return replace(
        base,
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("1.7B"),
        model_converters=_qwen3_1_7b_converters(),
    )


def sft_qwen3_1_7b_wordle() -> TrainerConfig:
    """SFT warmup for Qwen3-1.7B to play Wordle.

    Goal: 20 steps on ``willcb/V3-wordle`` to teach the model the
    environment's message format (multi-turn board-state → guess loop).

    Loads from HF base weights directly (no CPT prerequisite).
    Matches the prime-rl Wordle example SFT phase.

    Hardware: NGPU=1 (HF load safe), 64 GB NPU.
    """
    base = sft_default_config(_qwen3_1_7b_base())
    return replace(
        base,
        optimizer=replace(base.optimizer, lr=1e-5),
        lr_scheduler=replace(base.lr_scheduler, warmup_steps=0, decay_ratio=1.0, min_lr_factor=1.0),
        training=replace(base.training, local_batch_size=2, global_batch_size=64, seq_len=1024, max_norm=1.0, steps=20),
        dataloader=ChatDataLoaderConfig(
            dataset_path="willcb/V3-wordle",
            load_dataset_kwargs={"split": "train"},
            chat_processor="torchtitan_npu.hf_datasets.chat_processors.process_wordle_sample",
        ),
        checkpoint=replace(
            base.checkpoint,
            folder="checkpoint_wordle_sft",
            initial_load_in_hf=True,
            initial_load_path="./assets/hf/Qwen3-1.7B",
        ),
        activation_checkpoint=replace(base.activation_checkpoint, mode="selective"),
        profiling=replace(
            base.profiling,
            profile_ranks=[0],
            profile_step_start=6,
            profile_step_end=7,
            profile_with_memory=True,
        ),
    )
