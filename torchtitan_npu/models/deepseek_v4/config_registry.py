# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    ChatDataLoaderConfig,
    CheckpointConfig,
    OptimizerConfig,
    ParallelismConfig,
    TrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.patches.encoders.dsv4 import DSV4EncoderConfig

from . import model_registry


def _default_converters() -> list:
    return [
        # Migrated to the new ModelCustomConfig registry by upstream MR !144.
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_smla"),
        get_model_converter_config("npu_mhc_pre"),
    ]


def _enable_all_converters() -> list:
    return [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_smla"),
        get_model_converter_config("npu_mhc_pre"),
        get_model_converter_config("npu_mhc_post"),
    ]


def debug_deepseek_v4_flash_single_node() -> TrainerConfig:
    return TrainerConfig(
        model_spec=model_registry("v4_flash_debug_16_experts_43_layers"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=True),
        model_converters=ModelConvertersContainer.Config(converters=_enable_all_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=4,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            load_step=0,
            interval=500,
            last_save_model_only=True,
            load_only=True,
            initial_load_in_hf=False,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )


def debug_deepseek_v4_flash_single_node_mxfp8() -> TrainerConfig:
    return TrainerConfig(
        model_spec=model_registry("v4_flash_debug_16_experts_43_layers"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=True),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                *_enable_all_converters(),
                MXFP8Converter.Config(
                    recipe_name="mxfp8_rceil",
                    fqns=[
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
                    ],
                ),
            ]
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=4,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            load_step=0,
            interval=500,
            last_save_model_only=True,
            load_only=True,
            initial_load_in_hf=False,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )


def deepseek_v4_flash_4k_128die() -> TrainerConfig:
    return TrainerConfig(
        model_spec=model_registry("v4_flash_debug_256_experts_43_layers"),
        debug=DebugConfig(print_config=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=400,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            global_batch_size=1024,
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=64,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=True,
            folder="checkpoint",
            load_step=0,
            initial_load_in_hf=True,
            interval=10000,
            last_save_model_only=True,
            load_only=True,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )


def debug_deepseek_v4_pro_single_node() -> TrainerConfig:
    return TrainerConfig(
        model_spec=model_registry("v4_pro_debug_16_layers"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=4,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=16,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="checkpoint",
            load_step=0,
            interval=500,
            last_save_model_only=True,
            load_only=True,
            initial_load_in_hf=False,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )


def deepseek_v4_pro_4k_384die() -> TrainerConfig:
    return TrainerConfig(
        model_spec=model_registry("v4_pro_debug_61_layers"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=False),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=400,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            global_batch_size=384,
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=2000,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
            expert_parallel_degree=192,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=True,
            folder="checkpoint",
            load_step=0,
            initial_load_in_hf=True,
            interval=10000,
            last_save_model_only=True,
            load_only=True,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
    )


def debug_deepseek_v4_smoketest() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv4_tokenizer",
        model_spec=model_registry("smoketest"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=[]),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
            num_mtp_modules=0,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )


def sft_deepseek_v4_flash_16k_128die_tau() -> TrainerConfig:
    """SFT config for DeepSeek V4 flash debug (256 experts, 43 layers) on tau-bench-synthetic."""

    def process_tau_sample(sample):
        import json

        # tau-bench stores messages/tools as JSON strings in parquet
        raw_messages = sample["messages"]
        messages = json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages
        messages = [dict(m) for m in messages]

        raw_tools = sample.get("tools", [])
        tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools

        if tools:
            if messages and messages[0].get("role") == "system":
                messages[0] = dict(messages[0])
                messages[0]["tools"] = tools
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})

        return messages

    return TrainerConfig(
        model_spec=model_registry("v4_flash_debug_256_experts_43_layers"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=False),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=ChatDataLoaderConfig(
            load_dataset_kwargs={
                "data_files": "train-00000-of-00001.parquet",
                "split": "train",
            },
            sample_processor=process_tau_sample,
            chat_encoder=DSV4EncoderConfig(
                encoding_module_path="./assets/hf/DeepSeek-V4-Flash-Base/encoding/encoding_dsv4.py",
            ),
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=16384,
            max_norm=1.0,
            steps=100,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=64,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=4,
        ),
        checkpoint=CheckpointConfig(
            enable=True,
            folder="checkpoint",
            load_step=0,
            interval=500,
            last_save_model_only=True,
            load_only=False,
            initial_load_in_hf=True,
            export_dtype="bfloat16",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )


def sft_deepseek_v4_flash_1k_128die_gsm8k() -> TrainerConfig:
    """SFT config for DeepSeek V4 flash debug (256 experts, 43 layers) on GSM8K."""

    def process_gsm8k_sample(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    return TrainerConfig(
        model_spec=model_registry("v4_flash_debug_256_experts_43_layers"),
        debug=DebugConfig(print_config=True, moe_force_load_balance=False),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=ChatDataLoaderConfig(
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_gsm8k_sample,
            chat_encoder=DSV4EncoderConfig(
                encoding_module_path="./assets/hf/DeepSeek-V4-Flash-Base/encoding/encoding_dsv4.py",
            ),
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-5,
            eps=1e-6,
            swap_optimizer=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=1024,
            max_norm=1.0,
            steps=100,
            num_mtp_modules=1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=64,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=True,
            folder="checkpoint",
            load_step=0,
            interval=500,
            last_save_model_only=True,
            load_only=False,
            initial_load_in_hf=True,
            export_dtype="bfloat16",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
    )
