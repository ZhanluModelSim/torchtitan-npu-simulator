# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public DeepSeek V4.1 training recipes."""

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.optim import AdamW
from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu.config import CompileConfig, ModelCompileConfig, OptimizerConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager as NPUCheckpointManager
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor
from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import (
    _A5_PREQUANTIZE_FQNS_FULL,
    _DEFAULT_TEXT_DATA_FILES,
    _apply_lora,
    _flash_overrides,
    _make_trainer_config,
    _v41_optimizer_config,
    _vision_sft_dataloader_config,
)


def deepseek_v4_1_flash_cpt_4k_a3() -> TrainerEx.Config:
    """Build 128-rank Flash CPT on A3 at 4K context."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash",
        seq_len=4096,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=128,
            data_parallel_replicate_degree=1,
            expert_parallel_degree=128,
            context_parallel_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            fsdp_reshard_after_forward="always",
            enable_sequence_parallel=True,
        ),
    )
    config.training.num_tokens_per_train_step = 4194304
    config.debug.print_config = True
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(
        folder="/path/to/save_ckpt",
        initial_load_path="/path/to/init_load_ckpt",
        initial_load_in_hf=True,
        load_only=True,
    )
    config.override.imports = _flash_overrides(a5=False, fused_attention=False)
    config.extension.quantization.enable_quantized_training = False
    config.extension.quantization.recipe = "all_block_fp8"
    return config


def deepseek_v4_1_flash_cpt_4k_a5() -> TrainerEx.Config:
    """Build 128-rank Flash CPT on A5 at 4K context with block-FP8 training."""
    config = deepseek_v4_1_flash_cpt_4k_a3()
    config.override.imports = _flash_overrides(a5=True, fused_attention=True)
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.enable_fsdp_prequantize = True
    config.extension.quantization.fsdp_prequantize_fqns = list(_A5_PREQUANTIZE_FQNS_FULL)
    return config


def deepseek_v4_1_flash_qat_4k_a5() -> TrainerEx.Config:
    """Build 128-rank Flash QAT on A5 with MXFP4 and quantized sparse attention."""
    config = deepseek_v4_1_flash_cpt_4k_a5()
    config.override.imports = _flash_overrides(a5=True, fused_attention=True, qat=True)
    config.extension.quantization.enable_mxfp4_qat = True
    config.extension.quantization.li_quantization = "mxfp4"
    config.extension.quantization.enable_sparse_attention_quantization = True
    return config


def deepseek_v4_1_flash_lora_4k_a3() -> TrainerEx.Config:
    """Build 128-rank Flash LoRA on A3 at 4K context."""
    config = deepseek_v4_1_flash_cpt_4k_a3()
    # The LoRA transform revalidates the trainer and requires eager mode.
    config.compile = None
    config.checkpointer = None
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)])
    config = _apply_lora(config)
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.optim.optimizer = _v41_optimizer_config(config.model, include_engram_table=False)
    config.checkpointer = DeepSeekV41LoRACheckpointManager.Config(
        folder="/path/to/save_ckpt",
        initial_load_path="/path/to/init_load_ckpt",
        initial_load_in_hf=True,
        load_only=False,
    )
    config.override.imports = _flash_overrides(a5=False, fused_attention=False, swap_optimizer=False)
    return config


def deepseek_v4_1_flash_sft_4k_a3() -> TrainerEx.Config:
    """Build 128-rank multimodal Flash SFT on A3 at 4K context."""
    config = deepseek_v4_1_flash_cpt_4k_a3()
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.dataloader = _vision_sft_dataloader_config(config.model)
    assert config.checkpointer is not None
    config.checkpointer = derive(config.checkpointer, NPUCheckpointManager.Config)
    return config


def deepseek_v4_1_flash_sft_4k_a5() -> TrainerEx.Config:
    """Build 128-rank multimodal Flash SFT on A5 at 4K context with block-FP8 training."""
    config = deepseek_v4_1_flash_cpt_4k_a5()
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.dataloader = _vision_sft_dataloader_config(config.model)
    assert config.checkpointer is not None
    config.checkpointer = derive(config.checkpointer, NPUCheckpointManager.Config)
    return config


def deepseek_v4_1_flash_cpt_1m_a5() -> TrainerEx.Config:
    """Declare the long-context CP configuration; the model still rejects unsupported CP."""
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        data_parallel_replicate_degree=1,
        expert_parallel_degree=128,
        context_parallel_degree=128,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward="always",
        enable_sequence_parallel=True,
    )
    # TorchTitan validates the CP inner-attention types while the model config is
    # constructed, and the v0.4 CP lifecycle is not migrated yet. Assemble the
    # declaration on the CP=1 skeleton, then carry the requested topology on the
    # trainer config; parallelize rejects it through
    # ``validate_deepseek_v4_1_parallelism``.
    config = _make_trainer_config(
        "deepseek_v4_1_flash_text",
        parallelism=replace(parallelism, context_parallel_degree=1),
        seq_len=1048576,
        engram_enabled=False,
    )
    config.parallelism = parallelism
    config.training.num_tokens_per_train_step = 1048576
    config.debug.print_config = True
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    packed = config.dataloader.dataset
    assert isinstance(packed, ConcatThenSplitPackingConfig)
    dataset = packed.dataset
    assert isinstance(dataset, SingleDatasetConfig)
    assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
    assert isinstance(dataset.processor, AlignedTextProcessor.Config)
    dataset.source.load_dataset_kwargs["data_files"] = _DEFAULT_TEXT_DATA_FILES
    dataset.processor.per_doc_alignment = 256
    config.override.imports = _flash_overrides(a5=True, fused_attention=True, vision_rope=False)
    return config
