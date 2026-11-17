# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import fields
from importlib import import_module
from typing import TYPE_CHECKING, Any

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor

from torchtitan_npu.config import (
    CompileConfig,
    ModelCompileConfig,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor, compression_alignment
from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4.chat_encoder import DSV4EncoderConfig
from torchtitan_npu.models.deepseek_v4.chat_processor import DeepSeekV4ChatProcessor
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper
from torchtitan_npu.models.deepseek_v4.muon import _dsv4_muon_profile, _dsv4_optimizer_config

if TYPE_CHECKING:
    from torchtitan.config.transform import ModelConfigConverter
    from torchtitan.trainer import Trainer


def _per_doc_alignment(model: DeepSeekV4Model.Config) -> int:
    """The pooling granularity the loader pads every packed document to (V4: 128)."""
    alignment = compression_alignment(model.compress_ratios)
    return int(os.environ.get("DSV4_PER_DOC_ALIGNMENT", alignment))


def _make_trainer_config(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    parallelism = parallelism or ParallelismConfig(expert_parallel_degree=1, fsdp_reshard_after_forward="always")
    model = model_registry(
        flavor,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    model.local_compile_regions = []
    if num_mtp_layers > 0:
        loss = MTPChunkedLossWrapper.Config(
            mtp_scale=0.3,
            mtp_isolate_gradients=mtp_isolate_gradients,
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        )
    else:
        loss = ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        )
    config = TrainerEx.Config(
        hf_assets_path=hf_assets_path or "tests/assets/tokenizer",
        loss=loss,
        profiler=CANNProfiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model,
        dataloader=GrainDataLoader.Config(
            num_mtp_layers=num_mtp_layers,
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json", split="train", load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"}
                    ),
                    processor=AlignedTextProcessor.Config(per_doc_alignment=_per_doc_alignment(model)),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        optim=OptimConfig(
            optimizer=_dsv4_optimizer_config(model, lr=1e-5),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=20,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.01,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * model.max_context_length,
            num_tokens_per_train_step=-1,
            max_context_length=model.max_context_length,
            steps=100,
        ),
        parallelism=parallelism,
        activation_remat=None,
        compile=None,
        checkpointer=None,
    )
    config.override.imports = ["torchtitan_npu.override.common.rope.workaround"]
    return config


def deepseek_v4_debugmodel(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 2048,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    config = _make_trainer_config(
        "debugmodel",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.hf_assets_path = hf_assets_path or "tests/assets/deepseek_v3"
    return config


def deepseek_v4_flash(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_flash",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_debugmodel_npu(
    *,
    num_mtp_layers: int = 0,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 2048,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Construct native V4; NPU kernels are selected through component overrides."""
    return deepseek_v4_debugmodel(
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_flash_npu(
    *,
    num_mtp_layers: int = 0,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Construct native Flash; NPU kernels are selected through component overrides."""
    return _make_trainer_config(
        "deepseek_v4_flash",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_flash_43layers_16experts_npu(
    *,
    num_mtp_layers: int = 0,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Construct the reduced native Flash model for component overrides."""
    return _make_trainer_config(
        "deepseek_v4_flash_43layers_16experts",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_flash_lora(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    from torchtitan.config.transform import apply_transforms

    from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform
    from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager

    config = apply_transforms(
        deepseek_v4_flash(
            parallelism=parallelism,
            seq_len=seq_len,
            hf_assets_path=hf_assets_path,
            mtp_isolate_gradients=mtp_isolate_gradients,
        ),
        [deepseek_v4_lora_transform()],
    )
    assert isinstance(config.model, DeepSeekV4Model.Config)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    config.optim.optimizer._muon_profile = _dsv4_muon_profile(config.model)
    config.checkpointer = DeepSeekV4PEFTCheckpointManager.Config()
    return config


def deepseek_v4_flash_cp(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    raise NotImplementedError("Context parallelism is deferred in the v0.4 eager upgrade.")


def deepseek_v4_flash_sft(
    *,
    dataset_path: str | None = None,
    flavor: str = "deepseek_v4_flash",
    seq_len: int = 4096,
    num_packing_bins: int = 1,
    chat_encoder: DSV4EncoderConfig | None = None,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    if dataset_path is None:
        dataset_path = os.environ.get("DSV4_SFT_DATASET", "/path/to/train.jsonl")
    if not dataset_path:
        raise ValueError("SFT requires a non-empty dataset_path")
    config = _make_trainer_config(
        flavor,
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    assert isinstance(config.model, DeepSeekV4Model.Config)
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json", split="train", load_dataset_kwargs={"data_files": dataset_path}
                ),
                processor=DeepSeekV4ChatProcessor.Config(
                    per_doc_alignment=_per_doc_alignment(config.model),
                    chat_encoder=DSV4EncoderConfig() if chat_encoder is None else chat_encoder,
                ),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=num_packing_bins,
        ),
    )
    return config


def deepseek_v4_debugmodel_sft(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 2048,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return deepseek_v4_flash_sft(
        flavor="debugmodel",
        seq_len=seq_len,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_flash_43layers_16experts_sft(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return deepseek_v4_flash_sft(
        flavor="deepseek_v4_flash_43layers_16experts",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_flash_43layers_16experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_flash_43layers_16experts",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def _torchft_flash_config(
    base: Trainer.Config,
    *,
    expert_parallel_degree: int,
    data_parallel_shard_degree: int = 64,
    global_batch_size: int = 1024,
) -> TrainerEx.Config:
    # Check the optional dependency before importing upstream TorchFT modules.
    import_module("torchtitan_npu.extensions.torchtitan.experiments.torchft")

    from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager

    from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import (
        TorchFTOptimEx,
        TorchFTOptimizersContainerEx,
    )
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.trainer import FaultTolerantTrainerEx

    def config_values(source: object, target: type) -> dict[str, Any]:
        return {
            item.name: getattr(source, item.name) for item in fields(target) if item.init and hasattr(source, item.name)
        }

    values = config_values(base, FaultTolerantTrainerEx.Config)
    checkpoint_values = config_values(base.checkpointer, TorchFTCheckpointManager.Config)
    checkpoint_values.update(enable_ft_dataloader_checkpoints=True)
    values["checkpointer"] = TorchFTCheckpointManager.Config(**checkpoint_values)
    optim_values = config_values(base.optim, TorchFTOptimEx.Config)
    optim_values["optimizer"] = TorchFTOptimizersContainerEx.Config(
        **config_values(base.optim.optimizer, TorchFTOptimizersContainerEx.Config)
    )
    values["optim"] = TorchFTOptimEx.Config(**optim_values)
    config = FaultTolerantTrainerEx.Config(**values)
    assert isinstance(config.model, DeepSeekV4Model.Config)
    config.fault_tolerance.group_size = 2
    config.fault_tolerance.min_replica_size = 2
    config.training.num_tokens_per_train_step = global_batch_size * config.model.max_context_length
    config.training.disable_cuda_graphs = True
    if (
        config.parallelism.expert_parallel_degree != expert_parallel_degree
        or config.parallelism.data_parallel_shard_degree != data_parallel_shard_degree
    ):
        raise ValueError("Construct the base model with the TorchFT replica topology")
    return config


def deepseek_v4_flash_torchft() -> TrainerEx.Config:
    """Full Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(
        deepseek_v4_flash(
            parallelism=ParallelismConfig(
                expert_parallel_degree=64, data_parallel_shard_degree=64, fsdp_reshard_after_forward="always"
            )
        ),
        expert_parallel_degree=64,
    )


def deepseek_v4_flash_43layers_16experts_torchft() -> TrainerEx.Config:
    """The 16-expert Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(
        deepseek_v4_flash_43layers_16experts(
            parallelism=ParallelismConfig(
                expert_parallel_degree=16, data_parallel_shard_degree=64, fsdp_reshard_after_forward="always"
            )
        ),
        expert_parallel_degree=16,
    )


def deepseek_v4_flash_8p_torchft() -> TrainerEx.Config:
    """Four Flash layers plus MTP on two EP4/FSDP4 replicas for 8-card validation."""
    base = deepseek_v4_flash_43layers_16experts(
        num_mtp_layers=1,
        seq_len=256,
        parallelism=ParallelismConfig(
            expert_parallel_degree=4, data_parallel_shard_degree=4, fsdp_reshard_after_forward="always"
        ),
    )
    assert base.model is not None
    model = base.model
    assert isinstance(model, DeepSeekV4Model.Config)
    model.layers = model.layers[:4]
    model.n_layers = 4
    model.compress_ratios = model.compress_ratios[:4]
    config = _torchft_flash_config(
        base,
        expert_parallel_degree=4,
        data_parallel_shard_degree=4,
        global_batch_size=16,
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    config.debug.seed = 42
    return config


def deepseek_v4_pro(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_pro",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_pro_61layers_32experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_pro_61layers_32experts",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


def deepseek_v4_pro_61layers_12experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_pro_61layers_12experts",
        local_batch_size=1,
        seq_len=seq_len,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )


# --- GraphTrainer config factories ---


def deepseek_v4_flash_cpt(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_flash_43layers_16experts_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_43layers_16experts(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_flash_sft_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_sft(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_flash_43layers_16experts_sft_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_43layers_16experts_sft(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_flash_lora_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_lora(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_flash_cp_cpt(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_cp(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_pro_61layers_32experts_cpt(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_pro_61layers_32experts(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_pro_61layers_12experts_cpt(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    mtp_isolate_gradients: bool = False,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_pro_61layers_12experts(
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        mtp_isolate_gradients=mtp_isolate_gradients,
    )
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config
