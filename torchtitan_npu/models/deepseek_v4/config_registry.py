# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections.abc import Callable
from dataclasses import fields
from importlib import import_module
from typing import Any, cast

from torch.distributed.tensor import Shard
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.configs import (
    to_graph_trainer_config as _to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from torchtitan_npu.config import (
    MuonOptimizerProfile,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
    functionalize_recompute_mutations as _functionalize_mutations,
)

from . import (
    memory_policy,  # noqa: F401
    model_registry,
)
from .lora import DEEPSEEK_V4_LORA_TARGETS, DeepSeekV4LoRAConverter
from .model import DeepSeekV4Model, GraphTrainerDeepSeekV4Model
from .mtp import MTPChunkedLossWrapper
from .parallelize import parallelize_graph_trainer_deepseek_v4


def _lora_muon_pattern(compute_sharding_by_fqn):
    # AC/compile wrappers can insert prefixes before layer-local parameter names.
    suffixes = {fqn.split(".", 2)[-1] for fqn in compute_sharding_by_fqn if "lora_" in fqn}
    return "|".join(re.escape(suffix) for suffix in sorted(suffixes))


def _dsv4_muon_profile(model_spec: ModelSpec) -> MuonOptimizerProfile:
    """Build the DSV4-owned parameter and FlexShard policy for Muon."""
    model_config = cast("DeepSeekV4Model.Config", model_spec.model)
    # FSDP folds the (dp_shard, cp) storage mesh into the single
    # ``dp_shard_cp`` axis when context parallelism is enabled, and the
    # GraphTrainer path (simple_fsdp + partial_dtensor) stores dense
    # parameters on the ``fsdp`` axis.  ComputeLayout resolution filters
    # declarations by the parameter's actual storage mesh, so keep every
    # axis name to cover CP/non-CP, eager and GraphTrainer runs.
    dense_dp_axes = (
        MeshAxisName.DP_SHARD.value,
        f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}",
        MeshAxisName.FSDP.value,
    )
    owned = ComputeLayout(
        shardings_by_mesh_axis={axis: Owned() for axis in dense_dp_axes},
    )
    attention_shardings = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_per_head_projections = ("wq_b", "wo_a")
    attention_projections = tuple(attention_shardings) + attention_per_head_projections
    expert_projections = ("w1", "w2", "w3")
    routed_expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")
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
            {f"{prefix}.moe.shared_experts.{projection}.weight": owned for projection in expert_projections}
        )
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head flattened
        # into [n_heads * head_dim, q_lora_rank]; ``wo_a`` stores one
        # [o_lora_rank, per_group_in] matrix per group flattened likewise.
        # BlockShard computes Muon on each per-head/per-group matrix.
        shardings[f"{prefix}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={axis: BlockShard(dim=0, block_size=attention.head_dim) for axis in dense_dp_axes},
        )
        shardings[f"{prefix}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_size=attention.wo_a.out_features) for axis in dense_dp_axes
            },
        )
        if getattr(layer_config.attention, "compressor", None) is not None:
            for projection in ("wkv", "wgate"):
                shardings[f"{prefix}.attention.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.compressor.ape"] = owned
        expert_sharding = ComputeLayout(
            shardings_by_mesh_axis={
                **{axis: Shard(0) for axis in dense_dp_axes},
                MeshAxisName.EFSDP.value: Shard(0),
                MeshAxisName.EP.value: Shard(0),
            },
            shard_order_by_tensor_dim={  # pyrefly: ignore [unexpected-keyword]
                0: (MeshAxisName.EP.value, MeshAxisName.EFSDP.value),
            },
        )
        for projection in routed_expert_projections:
            shardings[f"{prefix}.moe.routed_experts.inner_experts.{projection}"] = expert_sharding
        shardings[f"{prefix}.moe.router.gate.weight"] = owned
        for module in hc_pre_modules:
            shardings[f"{prefix}.{module}.hc_fn"] = owned
        indexer = layer_config.attention.indexer
        if indexer is not None:
            shardings[f"{prefix}.attention.indexer.wq_b.weight"] = ComputeLayout(
                shardings_by_mesh_axis={
                    axis: BlockShard(
                        dim=0,
                        block_size=indexer.index_head_dim,
                    )
                    for axis in dense_dp_axes
                },
            )
            shardings[f"{prefix}.attention.indexer.weights_proj.weight"] = owned
            for projection in compressor_projections:
                shardings[f"{prefix}.attention.indexer.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.indexer.compressor.ape"] = owned
        if include_mtp_projections:
            for projection in mtp_projections:
                shardings[f"{prefix}.{projection}.weight"] = owned
            shardings[f"{prefix}.hc_head.hc_fn"] = owned
        for base_key, layout in tuple(shardings.items()):
            if base_key.endswith(".weight"):
                owner = base_key.removesuffix(".weight")
                if owner.removeprefix(f"{prefix}.") in DEEPSEEK_V4_LORA_TARGETS:
                    shardings[f"{owner}.lora_a.weight"] = owned
                    shardings[f"{owner}.lora_b.weight"] = layout
        for projection in ("w13_lora_a", "w1_lora_b", "w3_lora_b", "w2_lora_a", "w2_lora_b"):
            shardings[f"{prefix}.moe.routed_experts.inner_experts.{projection}"] = expert_sharding
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
        for layer_id, layer_config in enumerate(model_config.mtp_layers)
    )
    per_layer = main_layer_shardings + mtp_layer_shardings
    compute_sharding_by_fqn = {
        fqn: compute_sharding for layer_shardings in per_layer for fqn, compute_sharding in layer_shardings.items()
    }
    compute_sharding_by_fqn["hc_head.hc_fn"] = owned
    routed_expert_marker = ".moe.routed_experts.inner_experts."

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
    lora_pattern = _lora_muon_pattern(compute_sharding_by_fqn)
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.indexer\.(?:{'|'.join(indexer_projections)})\.weight|"
        rf"attention\.(?:compressor|indexer\.compressor)\.(?:{'|'.join(compressor_projections)})\.weight|"
        r"attention\.(?:compressor|indexer\.compressor)\.ape|"
        rf"moe\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"moe\.routed_experts\.inner_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"moe\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn|"
        rf"(?:{'|'.join(mtp_projections)})\.weight|"
        r"hc_head\.hc_fn" + (f"|{lora_pattern}" if lora_pattern else "") + r")$"
    )
    return MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": bucket_configs,
            }
        },
    )


def _dsv4_optimizer_config(
    model_spec: ModelSpec,
    *,
    lr: float,
) -> OptimizerConfig:
    """Build an AdamW-default DSV4 optimizer schema with a Muon profile.

    ``name`` remains ``AdamW`` until the user explicitly supplies
    ``--optimizer.name Muon``.  Keeping the profile on the ordinary recipe
    makes the CLI selection the sole Muon entry point.
    """
    adamw = default_adamw(lr=lr, eps=1e-6)
    return OptimizerConfig(
        lr=lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=10,
        muon_adjust_lr_fn="match_rms_adamw",
        _muon_profile=_dsv4_muon_profile(model_spec),
        param_groups=adamw.param_groups,
        implementation=adamw.implementation,
        optimizer_factory_kwargs_by_name=adamw.optimizer_factory_kwargs_by_name,
    )


def _make_trainer_config(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Trainer.Config:
    model_spec = model_registry(flavor, num_mtp_layers=num_mtp_layers, converters=converters)
    if num_mtp_layers > 0:
        loss = MTPChunkedLossWrapper.Config(
            mtp_scale=0.3,
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        )
    else:
        loss = ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
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
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=_dsv4_optimizer_config(model_spec, lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=100,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            fsdp_reshard_after_forward="always",
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(  # pyrefly: ignore[bad-argument-type]
            enable=False,
            interval=100,
        ),
    )
    if getattr(model_spec.model, "lora", None) is not None:
        from .peft import DeepSeekV4PEFTCheckpointManager

        config.checkpoint = DeepSeekV4PEFTCheckpointManager.Config(
            **{item.name: getattr(config.checkpoint, item.name) for item in fields(config.checkpoint) if item.init},
        )
    return config


def deepseek_v4_debugmodel(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Trainer.Config:
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
) -> Trainer.Config:
    return _make_trainer_config(
        "deepseek_v4_flash",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
        converters=converters,
    )


def deepseek_v4_flash_lora() -> Trainer.Config:
    return deepseek_v4_flash(
        converters=[DeepSeekV4LoRAConverter.Config(rank=16, alpha=32, rank_experts=16)],
    )


def deepseek_v4_flash_43layers_16experts(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Trainer.Config:
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
) -> Trainer.Config:
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
    checkpoint_values = config_values(base.checkpoint, TorchFTCheckpointManager.Config)
    checkpoint_values.update(enable=True, enable_ft_dataloader_checkpoints=True)
    values["checkpoint"] = TorchFTCheckpointManager.Config(**checkpoint_values)
    values["optimizer"] = TorchFTOptimizersContainerEx.Config(
        **config_values(base.optimizer, TorchFTOptimizersContainerEx.Config)
    )
    config = FaultTolerantTrainerEx.Config(**values)
    config.fault_tolerance.group_size = 2
    config.fault_tolerance.min_replica_size = 2
    config.training.global_batch_size = global_batch_size
    config.training.disable_cuda_graphs = True
    config.parallelism.expert_parallel_degree = expert_parallel_degree
    config.parallelism.data_parallel_shard_degree = data_parallel_shard_degree
    config.parallelism.spmd_backend = "spmd_types"
    return config


def deepseek_v4_flash_torchft() -> Trainer.Config:
    """Full Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(deepseek_v4_flash(), expert_parallel_degree=64)


def deepseek_v4_flash_43layers_16experts_torchft() -> Trainer.Config:
    """The 16-expert Flash model on two 64-rank TorchFT replicas."""
    return _torchft_flash_config(deepseek_v4_flash_43layers_16experts(), expert_parallel_degree=16)


def deepseek_v4_flash_8p_torchft() -> Trainer.Config:
    """Four Flash layers plus MTP on two EP4/FSDP4 replicas for 8-card validation."""
    base = deepseek_v4_flash_43layers_16experts(num_mtp_layers=1)
    assert base.model_spec is not None
    model = base.model_spec.model
    assert isinstance(model, DeepSeekV4Model.Config)
    model.layers = model.layers[:4]
    model.n_layers = 4
    model.compress_ratios = model.compress_ratios[:4]
    base.model_spec.flavor = "deepseek_v4_flash_4layers_16experts"
    config = _torchft_flash_config(
        base,
        expert_parallel_degree=4,
        data_parallel_shard_degree=4,
        global_batch_size=16,
    )
    config.training.seq_len = 256
    config.debug.seed = 42
    return config


def deepseek_v4_pro(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Trainer.Config:
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
) -> Trainer.Config:
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
) -> Trainer.Config:
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
    graph_model_registry: Callable[[str], ModelSpec],
) -> GraphTrainer.Config:
    """Project DSV4's NPU recipe config onto GraphTrainer's upstream schema."""
    if isinstance(base_config, TrainerEx.Config):
        base_config = Trainer.Config(
            **{config_field.name: getattr(base_config, config_field.name) for config_field in fields(Trainer.Config)}
        )
    return _to_graph_trainer_config(base_config, graph_model_registry)


def _graph_trainer_model_registry(flavor: str) -> ModelSpec:
    """Build a ModelSpec for the graph_trainer DeepSeek V4 path.

    Wraps the base model config in ``GraphTrainerDeepSeekV4Model.Config`` and
    points to ``parallelize_graph_trainer_deepseek_v4``. ``to_graph_trainer_config``
    re-wraps the base config's model fields into this Config class, so the
    values copied here only carry the class identity and parallelize_fn.
    """
    spec = model_registry(flavor)
    graph_model = GraphTrainerDeepSeekV4Model.Config(
        **{f.name: getattr(spec.model, f.name) for f in fields(spec.model)}
    )
    return ModelSpec(
        name=spec.name,
        flavor=flavor,
        model=graph_model,
        parallelize_fn=parallelize_graph_trainer_deepseek_v4,
        pipelining_fn=spec.pipelining_fn,
        post_optimizer_build_fn=spec.post_optimizer_build_fn,
        state_dict_adapter=spec.state_dict_adapter,
    )


def _graph_trainer_compile_config() -> GraphTrainerCompileConfig:
    """Compile settings shared by the GraphTrainer flash/pro config factories."""
    return GraphTrainerCompileConfig(
        enable=True,
        mode="aot_fx_trace",
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
    config = to_graph_trainer_config(
        deepseek_v4_debugmodel(num_mtp_layers=0),
        _graph_trainer_model_registry,
    )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        mode="aot_fx_trace",
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
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_flash_43layers_16experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash 43 Layers 16 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash_43layers_16experts(num_mtp_layers=0),
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro(num_mtp_layers=0),
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro_61layers_32experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro 61 Layers 32 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro_61layers_32experts(num_mtp_layers=0),
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config
