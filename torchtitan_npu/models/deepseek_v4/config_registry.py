# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    ChatDataLoaderConfig,
    TrainerConfig,
    cpt_default_config,
    debug_single_node_eq_pruned_config,
    sft_default_config,
    trainer_base_config,
)
from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.hf_datasets.chat_processors import dsv4_chat_encoder, process_gsm8k_sample, process_tau_sample

from . import model_registry

_MXFP8_FQNS = (
    "pre_attention.wq_a",
    "pre_attention.wq_b",
    "pre_attention.wkv",
    "pre_attention.indexer.wq_b",
    "pre_attention.indexer.weights_proj",
    "post_attention.wo_a",
    "post_attention.wo_b",
    "moe.experts",
    "moe.shared_experts",
    "e_proj",
    "h_proj",
)

_DEFAULT_CONVERTERS = ("npu_rms_norm", "npu_moe_dispatch", "npu_gmm", "npu_rope", "npu_smla", "npu_mhc_pre")


def _default_converters(*extra_names: str) -> list:
    return [get_model_converter_config(name) for name in (*_DEFAULT_CONVERTERS, *extra_names)]


def debug_deepseek_v4_smoketest() -> TrainerConfig:
    base = trainer_base_config()
    return replace(
        base,
        hf_assets_path="./tests/assets/tokenizer/deepseekv4_tokenizer",
        model_spec=model_registry("smoketest"),
        optimizer=replace(base.optimizer, swap_optimizer=False),
        lr_scheduler=replace(base.lr_scheduler, min_lr_factor=0.1, warmup_steps=2),
        training=replace(base.training, seq_len=128, steps=2, num_mtp_modules=0),
        activation_checkpoint=replace(base.activation_checkpoint, mode="none"),
    )


def _flash_base() -> TrainerConfig:
    base = trainer_base_config()
    return replace(
        base,
        model_spec=model_registry("v4_flash_debug_256_experts_43_layers"),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        training=replace(base.training, num_mtp_modules=1),
    )


def deepseek_v4_flash_4k_128die() -> TrainerConfig:
    base = cpt_default_config(_flash_base())
    return replace(
        base,
        training=replace(base.training, global_batch_size=1024),
        parallelism=replace(base.parallelism, data_parallel_shard_degree=128, expert_parallel_degree=64),
    )


def deepseek_v4_pro_4k_384die() -> TrainerConfig:
    base = replace(deepseek_v4_flash_4k_128die(), model_spec=model_registry("v4_pro_debug_61_layers"))
    return replace(
        base,
        training=replace(base.training, global_batch_size=384),
        parallelism=replace(base.parallelism, data_parallel_shard_degree=384, expert_parallel_degree=192),
        compile=replace(base.compile, enable=True),  # TODO check whether enable compile in base config
    )


def debug_deepseek_v4_flash_single_node() -> TrainerConfig:
    base = debug_single_node_eq_pruned_config(_flash_base())
    return replace(
        base,
        model_spec=model_registry("v4_flash_debug_16_experts_43_layers"),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters("npu_mhc_post")),
        parallelism=replace(base.parallelism, expert_parallel_degree=8),
    )


def debug_deepseek_v4_flash_single_node_mxfp8() -> TrainerConfig:
    base = debug_deepseek_v4_flash_single_node()
    return replace(
        base,
        model_converters=ModelConvertersContainer.Config(
            converters=base.model_converters.converters  # noqa: RUF005
            + [MXFP8Converter.Config(recipe_name="mxfp8_rceil", fqns=list(_MXFP8_FQNS))]
        ),
    )


def debug_deepseek_v4_pro_single_node() -> TrainerConfig:
    base = debug_deepseek_v4_flash_single_node()
    return replace(
        base,
        model_spec=model_registry("v4_pro_debug_16_layers"),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        parallelism=replace(base.parallelism, expert_parallel_degree=16),
    )


def sft_deepseek_v4_flash_16k_128die_tau() -> TrainerConfig:
    base = sft_default_config(deepseek_v4_flash_4k_128die())
    return replace(
        base,
        dataloader=ChatDataLoaderConfig(
            load_dataset_kwargs={"data_files": "train-00000-of-00001.parquet", "split": "train"},
            sample_processor=process_tau_sample,
            chat_encoder=dsv4_chat_encoder(),
        ),
        training=replace(base.training, seq_len=16384),
        parallelism=replace(base.parallelism, context_parallel_degree=4),
    )


def sft_deepseek_v4_flash_1k_128die_gsm8k() -> TrainerConfig:
    base = sft_default_config(deepseek_v4_flash_4k_128die())
    return replace(
        base,
        dataloader=ChatDataLoaderConfig(
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_gsm8k_sample,
            chat_encoder=dsv4_chat_encoder(),
        ),
        training=replace(base.training, seq_len=1024),
    )
