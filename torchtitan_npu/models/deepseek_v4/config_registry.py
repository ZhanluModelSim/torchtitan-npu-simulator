# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import fields
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import torch
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_npu.config import (
    CompileConfig,
    ModelCompileConfig,
    MuonOptimizerProfile,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor, compression_alignment
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts

from . import model_registry

if TYPE_CHECKING:
    from torchtitan.config.override import OverrideImport
    from torchtitan.config.transform import ModelConfigConverter
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from .chat_encoder import DSV4EncoderConfig
from .chat_processor import DeepSeekV4ChatProcessor
from .model import DeepSeekV4Model, GraphTrainerDeepSeekV4Model
from .mtp import MTPChunkedLossWrapper


def _dsv4_muon_profile(model: DeepSeekV4Model.Config) -> MuonOptimizerProfile:
    """Build the DSV4-owned parameter and FlexShard policy for Muon."""
    model_config = model
    dense_dp_axes = (MeshAxisName.DP_SHARD.value,)
    owned = make_owned_layout(dense_dp_axes)
    attention_shardings = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_per_head_projections = ("wq_b", "wo_a")
    attention_projections = tuple(attention_shardings) + attention_per_head_projections
    expert_projections = ("w13", "w2")
    routed_expert_projections = ("w13.weight", "w2.weight")
    compressor_projections = ("wkv", "wgate")
    indexer_projections = ("wq_b", "weights_proj")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    mtp_projections = ("e_proj", "h_proj")

    def compute_shardings_for_transformer_layer(
        prefix: str,
        layer_config: Any,
        *,
        include_mtp_projections: bool,
    ) -> dict[str, ComputeLayout]:
        attention = layer_config.attention
        shardings = {
            f"{prefix}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in attention_shardings.items()
        }
        shardings.update(
            {f"{prefix}.feed_forward.shared_experts.{projection}.weight": owned for projection in expert_projections}
        )
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head flattened
        # into [n_heads * head_dim, q_lora_rank]; ``wo_a`` stores one
        # [o_lora_rank, per_group_in] matrix per group flattened likewise.
        # BlockShard computes Muon on each per-head/per-group matrix.
        shardings[f"{prefix}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.head_dim,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        shardings[f"{prefix}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.wo_a.out_features,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        if getattr(layer_config.attention, "compressor", None) is not None:
            for projection in ("wkv", "wgate"):
                shardings[f"{prefix}.attention.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.compressor.ape"] = owned
        expert_sharding = make_expert_layout(dense_dp_axes)
        for projection in routed_expert_projections:
            shardings[f"{prefix}.feed_forward.routed_experts.{projection}"] = expert_sharding
        shardings[f"{prefix}.feed_forward.router.gate.weight"] = owned
        for module in hc_pre_modules:
            shardings[f"{prefix}.{module}.hc_fn"] = owned
        indexer = layer_config.attention.indexer
        if indexer is not None:
            shardings[f"{prefix}.attention.indexer.wq_b.weight"] = ComputeLayout(
                shardings_by_mesh_axis={
                    axis: BlockShard(
                        dim=0,
                        block_sizes=(indexer.index_head_dim,),
                    )
                    for axis in dense_dp_axes
                },
                compute_dtype=torch.bfloat16,
            )
            shardings[f"{prefix}.attention.indexer.weights_proj.weight"] = owned
            for projection in compressor_projections:
                shardings[f"{prefix}.attention.indexer.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.indexer.compressor.ape"] = owned
        if include_mtp_projections:
            for projection in mtp_projections:
                shardings[f"{prefix}.{projection}.weight"] = owned
            shardings[f"{prefix}.hc_head.hc_fn"] = owned
        return shardings

    main_layer_shardings = tuple(
        compute_shardings_for_transformer_layer(
            f"layers.{layer_id}",
            model_config.layers[layer_id],
            include_mtp_projections=False,
        )
        for layer_id in range(model_config.n_layers)
    )
    mtp_layer_shardings = tuple(
        compute_shardings_for_transformer_layer(
            f"mtp_layers.{layer_id}",
            layer_config,
            include_mtp_projections=True,
        )
        for layer_id, layer_config in enumerate(model_config.mtp_layers or ())
    )
    per_layer = main_layer_shardings + mtp_layer_shardings
    compute_sharding_by_fqn = {
        fqn: compute_sharding for layer_shardings in per_layer for fqn, compute_sharding in layer_shardings.items()
    }
    compute_sharding_by_fqn["hc_head.hc_fn"] = owned
    routed_expert_marker = ".feed_forward.routed_experts."

    def bucket_configs_for_layers(
        prefix: str,
        layer_shardings: tuple[dict[str, ComputeLayout], ...],
    ) -> tuple[BucketConfig, ...]:
        layer_bucket_configs = []
        for layer_id, shardings in enumerate(layer_shardings):
            layer_name = f"{prefix}.{layer_id}"
            routed_expert_patterns = tuple(fqn for fqn in shardings if routed_expert_marker in fqn)
            dense_patterns = tuple(fqn for fqn in shardings if routed_expert_marker not in fqn)
            layer_bucket_configs.append(BucketConfig(name=layer_name, patterns=dense_patterns))
            layer_bucket_configs.append(
                BucketConfig(
                    name=f"{layer_name}.routed-experts",
                    patterns=routed_expert_patterns,
                )
            )
        return tuple(layer_bucket_configs)

    bucket_configs = bucket_configs_for_layers("layers", main_layer_shardings)
    bucket_configs += bucket_configs_for_layers("mtp_layers", mtp_layer_shardings)
    # hc_head is global rather than layer-scoped, so it needs its own bucket.
    bucket_configs += (BucketConfig(name="hc_head", patterns=("hc_head.hc_fn",)),)
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.indexer\.(?:{'|'.join(indexer_projections)})\.weight|"
        rf"attention\.(?:compressor|indexer\.compressor)\.(?:{'|'.join(compressor_projections)})\.weight|"
        r"attention\.(?:compressor|indexer\.compressor)\.ape|"
        rf"feed_forward\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"feed_forward\.routed_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"feed_forward\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn|"
        rf"(?:{'|'.join(mtp_projections)})\.weight|"
        r"hc_head\.hc_fn" + r")$"
    )
    from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon

    profile = MuonOptimizerProfile(
        optimizer_config_type=DistMuon.Config,
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": bucket_configs,
            }
        },
    )
    return with_lora_compute_layouts(model, profile)


def _dsv4_optimizer_config(
    model: DeepSeekV4Model.Config,
    *,
    lr: float,
) -> OptimizerConfig:
    """Build recipe-owned AdamW and Muon + AdamW defaults for DSV4.

    ``--optim.optimizer.name=Muon`` selects the mixed optimizer defaults.
    Non-None scalar CLI values override the selected mode's defaults;
    the profile supplies Muon parameter matching and FlexShard layouts.
    """
    profile = _dsv4_muon_profile(model)
    return OptimizerConfig(
        _optimizer_defaults={
            "AdamW": [AdamW.Config(pattern=r".*", lr=lr, eps=1e-6)],
            "Muon": [
                profile.optimizer_config_type(
                    pattern=profile.muon_pattern,
                    lr=lr,
                    ns_steps=10,
                    adjust_lr_fn="match_rms_adamw",
                    **profile.optimizer_factory_kwargs["DistMuon"],
                ),
                AdamW.Config(pattern=r".*", lr=lr, eps=1e-8, fused=True, foreach=False),
            ],
        },
        _muon_profile=profile,
    )


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
) -> TrainerEx.Config:
    model = model_registry(flavor, seq_len=seq_len, num_mtp_layers=num_mtp_layers, converters=converters)
    if num_mtp_layers > 0:
        loss = MTPChunkedLossWrapper.Config(
            mtp_scale=0.3,
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            fsdp_reshard_after_forward="always",
        ),
        activation_remat=None,
        compile=None,
        checkpointer=None,
    )
    return config


def deepseek_v4_debugmodel(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "debugmodel",
        local_batch_size=1,
        seq_len=2048,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
    )


def deepseek_v4_flash(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_flash",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
    )


def deepseek_v4_debugmodel_npu(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    """Construct native V4; NPU kernels are selected through component overrides."""
    return _make_trainer_config("debugmodel", local_batch_size=1, seq_len=2048, num_mtp_layers=num_mtp_layers)


def deepseek_v4_flash_npu(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    """Construct native Flash; NPU kernels are selected through component overrides."""
    return _make_trainer_config("deepseek_v4_flash", local_batch_size=1, seq_len=4096, num_mtp_layers=num_mtp_layers)


def deepseek_v4_flash_43layers_16experts_npu(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    """Construct the reduced native Flash model for component overrides."""
    return _make_trainer_config(
        "deepseek_v4_flash_43layers_16experts",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
    )


def deepseek_v4_flash_lora() -> TrainerEx.Config:
    from torchtitan.config.transform import apply_transforms

    from .lora import deepseek_v4_lora_transform
    from .peft import DeepSeekV4PEFTCheckpointManager

    config = apply_transforms(deepseek_v4_flash(), [deepseek_v4_lora_transform()])
    assert isinstance(config.model, DeepSeekV4Model.Config)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    config.optim.optimizer._muon_profile = _dsv4_muon_profile(config.model)
    config.checkpointer = DeepSeekV4PEFTCheckpointManager.Config()
    return config


def deepseek_v4_flash_cp() -> TrainerEx.Config:
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


def deepseek_v4_debugmodel_sft() -> TrainerEx.Config:
    return deepseek_v4_flash_sft(flavor="debugmodel", seq_len=2048)


def deepseek_v4_flash_43layers_16experts_sft() -> TrainerEx.Config:
    return deepseek_v4_flash_sft(flavor="deepseek_v4_flash_43layers_16experts")


def deepseek_v4_flash_43layers_16experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_flash_43layers_16experts",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
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

    from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainerEx
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.trainer import FaultTolerantTrainerEx

    def config_values(source: object, target: type) -> dict[str, Any]:
        return {
            item.name: getattr(source, item.name) for item in fields(target) if item.init and hasattr(source, item.name)
        }

    values = config_values(base, FaultTolerantTrainerEx.Config)
    checkpoint_values = config_values(base.checkpointer, TorchFTCheckpointManager.Config)
    checkpoint_values.update(enable_ft_dataloader_checkpoints=True)
    values["checkpointer"] = TorchFTCheckpointManager.Config(**checkpoint_values)
    values["optim"] = OptimConfig(
        optimizer=TorchFTOptimizersContainerEx.Config(
            **config_values(base.optim.optimizer, TorchFTOptimizersContainerEx.Config)
        ),
        lr_scheduler=base.optim.lr_scheduler,
        max_norm=base.optim.max_norm,
    )
    config = FaultTolerantTrainerEx.Config(**values)
    assert isinstance(config.model, DeepSeekV4Model.Config)
    config.fault_tolerance.group_size = 2
    config.fault_tolerance.min_replica_size = 2
    config.training.num_tokens_per_train_step = global_batch_size * config.model.max_context_length
    config.training.disable_cuda_graphs = True
    config.parallelism.expert_parallel_degree = expert_parallel_degree
    config.parallelism.data_parallel_shard_degree = data_parallel_shard_degree
    return config


def deepseek_v4_flash_torchft() -> TrainerEx.Config:
    """Full Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(deepseek_v4_flash(), expert_parallel_degree=64)


def deepseek_v4_flash_43layers_16experts_torchft() -> TrainerEx.Config:
    """The 16-expert Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(deepseek_v4_flash_43layers_16experts(), expert_parallel_degree=16)


def deepseek_v4_flash_8p_torchft() -> TrainerEx.Config:
    """Four Flash layers plus MTP on two EP4/FSDP4 replicas for 8-card validation."""
    base = deepseek_v4_flash_43layers_16experts(num_mtp_layers=1)
    assert base.model is not None
    model = base.model
    assert isinstance(model, DeepSeekV4Model.Config)
    model.layers = model.layers[:4]
    model.n_layers = 4
    model.compress_ratios = model.compress_ratios[:4]
    # The token budget is derived from the model context, so shrink the context
    # before the TorchFT config computes it.
    model.max_context_length = 256
    base.training.max_context_length = 256
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
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_pro",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
    )


def deepseek_v4_pro_61layers_32experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_pro_61layers_32experts",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
    )


def deepseek_v4_pro_61layers_12experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_pro_61layers_12experts",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
    )


# --- GraphTrainer config factories ---


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_config_cls: type[DeepSeekV4Model.Config],
) -> GraphTrainer.Config:
    """Project DSV4's NPU recipe config onto GraphTrainer's upstream schema."""
    if isinstance(base_config, TrainerEx.Config):
        base_config = Trainer.Config(
            **{config_field.name: getattr(base_config, config_field.name) for config_field in fields(Trainer.Config)}
        )
    from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config as _to_graph_trainer_config

    return _to_graph_trainer_config(base_config, model_config_cls)


def _graph_trainer_compile_config() -> GraphTrainerCompileConfig:
    """Compile settings shared by the GraphTrainer flash/pro config factories."""
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations as _functionalize_mutations,
    )

    return GraphTrainerCompileConfig(
        memory_policy="full",
        # Keeps recomputed forward mutations (fused partial RoPE) faithful;
        # see the functionalize_recompute_mutations patch module.
        pass_pipeline=_functionalize_mutations.PASS_PIPELINE_NAME,
        disable_passes=[
            "cudagraph_pass",
        ],
    )


def graph_trainer_deepseek_v4_debugmodel() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 debug model"""
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations as _functionalize_mutations,
    )

    config = to_graph_trainer_config(
        deepseek_v4_debugmodel(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = GraphTrainerCompileConfig(
        memory_policy="full",
        enable_passes=True,
        pass_pipeline=_functionalize_mutations.PASS_PIPELINE_NAME,
        disable_passes=[
            "cudagraph_pass",
        ],
    )
    return config


def graph_trainer_deepseek_v4_flash() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_flash_43layers_16experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash 43 Layers 16 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash_43layers_16experts(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro_61layers_32experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro 61 Layers 32 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro_61layers_32experts(num_mtp_layers=0),
        GraphTrainerDeepSeekV4Model.Config,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def deepseek_v4_flash_cpt() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_flash_8p_cpt_4k_a3() -> TrainerEx.Config:
    """A3 eight-card eager recipe for the 43-layer, 16-expert Flash model."""
    config = deepseek_v4_flash_43layers_16experts()
    if os.environ.get("DSV4_A3_USE_GOLDEN", "0") == "1":
        config.model.local_compile_regions = []
    max_context_length = int(os.environ.get("DSV4_A3_MAX_CONTEXT_LENGTH", "4096"))
    cast("DeepSeekV4Model.Config", config.model).max_context_length = max_context_length
    config.training.max_context_length = max_context_length
    if os.environ.get("DSV4_A3_USE_GOLDEN", "0") == "1":
        from torchtitan.distributed.activation_remat import FullActivationRematConfig

        config.activation_remat = FullActivationRematConfig()
    # Avoid the NPU Nonzero kernel in the variable-length metadata builder.
    # The document-aware packer and fixed-capacity metadata use cumsum/where
    # instead and preserve the same packed document boundaries.
    config.dataloader.max_num_documents = int(os.environ.get("DSV4_A3_MAX_NUM_DOCUMENTS", "1"))

    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", config.hf_assets_path)
    config.parallelism.data_parallel_replicate_degree = int(os.environ.get("DSV4_A3_DP_REPLICATE", "1"))
    config.parallelism.data_parallel_shard_degree = int(os.environ.get("DSV4_A3_DP_SHARD", "8"))
    config.parallelism.expert_parallel_degree = int(os.environ.get("DSV4_A3_EP", "8"))
    config.parallelism.tensor_parallel_degree = int(os.environ.get("DSV4_A3_TP", "1"))
    config.parallelism.context_parallel_degree = int(os.environ.get("DSV4_A3_CP", "1"))
    config.parallelism.pipeline_parallel_degree = int(os.environ.get("DSV4_A3_PP", "1"))

    config.training.num_tokens_per_microbatch_per_dp_rank = int(os.environ.get("DSV4_A3_TOKENS_PER_MICROBATCH", "4096"))
    config.training.num_tokens_per_train_step = int(os.environ.get("DSV4_A3_TOKENS_PER_STEP", "262144"))
    config.training.steps = int(os.environ.get("DSV4_A3_STEPS", "100"))
    config.training.disable_cuda_graphs = True
    if os.environ.get("DSV4_A3_SEED"):
        config.debug.seed = int(os.environ["DSV4_A3_SEED"])
    config.debug.deterministic = os.environ.get("DSV4_A3_DETERMINISTIC", "0") == "1"
    config.comm.init_timeout_seconds = int(os.environ.get("DSV4_A3_INIT_TIMEOUT", "7200"))
    config.comm.train_timeout_seconds = int(os.environ.get("DSV4_A3_TRAIN_TIMEOUT", "600"))

    optimizer = cast("OptimizerConfig", config.optim.optimizer)
    optimizer.name = cast(
        "Any",
        os.environ.get(
            "DSV4_A3_OPTIMIZER",
            "AdamW" if os.environ.get("DSV4_A3_USE_GOLDEN", "0") == "1" else "Muon",
        ),
    )
    optimizer.lr = float(os.environ.get("DSV4_A3_LR", "1.0e-5"))
    optimizer.beta1 = float(os.environ.get("DSV4_A3_BETA1", "0.9"))
    optimizer.beta2 = float(os.environ.get("DSV4_A3_BETA2", "0.95"))
    optimizer.eps = float(os.environ.get("DSV4_A3_EPS", "1.0e-8"))
    optimizer.weight_decay = float(os.environ.get("DSV4_A3_WEIGHT_DECAY", "1.0e-1"))
    optimizer.muon_momentum = float(os.environ.get("DSV4_A3_MUON_MOMENTUM", "0.95"))
    optimizer.muon_enable_nesterov = True
    optimizer.muon_ns_steps = int(os.environ.get("DSV4_A3_MUON_NS_STEPS", "10"))
    optimizer.muon_adjust_lr_fn = cast("Any", os.environ.get("DSV4_A3_MUON_ADJUST_LR_FN", "match_rms_adamw"))
    optimizer.materialize()

    imports: list[OverrideImport]
    if os.environ.get("DSV4_A3_USE_GOLDEN", "0") == "1":
        imports = ["torchtitan_npu.override.common.rope.workaround"]
    else:
        imports = [
            "torchtitan_npu.override.common.rms_norm.asc",
            (
                "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
                {"selector_backend": "asc"},
            ),
            "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
            "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
            "torchtitan_npu.override.common.token_dispatcher.asc",
        ]
        if "torchtitan_npu.override.deepseek_v4.compressor.asc" in os.environ.get("DSV4_A3_EXTRA_OVERRIDES", ""):
            imports[0] = "torchtitan_npu.override.deepseek_v4.common.rms_norm"
    if os.environ.get("DSV4_A3_USE_GOLDEN", "0") != "1":
        imports.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    extra_overrides = os.environ.get("DSV4_A3_EXTRA_OVERRIDES", "").split()
    imports.extend(entry for entry in extra_overrides if entry != "torchtitan_npu.override.deepseek_v4.compressor.asc")
    config.override.imports = imports
    return config


def deepseek_v4_flash_43layers_16experts_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_43layers_16experts()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_flash_sft_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_flash_43layers_16experts_sft_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_43layers_16experts_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_flash_lora_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_lora()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_flash_cp_cpt() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_flash_cp()
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_pro_61layers_32experts_cpt() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_pro_61layers_32experts()
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_pro_61layers_12experts_cpt() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_pro_61layers_12experts()
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def graph_trainer_deepseek_v4_flash_cpt() -> GraphTrainer.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = graph_trainer_deepseek_v4_flash()
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config
