# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import fields
from typing import Any, cast

from torch.distributed.tensor import Shard
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    AttentionPerHeadComputeView,
    BucketConfig,
    ComputeLayout,
    MuonComputeShardingConfig,
    Owned,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.configs import (
    to_graph_trainer_config as _to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_npu.config import (
    MuonOptimizerProfile,
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
)

from . import (
    memory_policy,  # noqa: F401
    model_registry,
)
from .model import DeepSeekV4Model, GraphTrainerDeepSeekV4Model
from .mtp import MTPChunkedLossWrapper
from .parallelize import parallelize_graph_trainer_deepseek_v4


def _dsv4_muon_profile(model_spec: ModelSpec) -> MuonOptimizerProfile:
    """Build the DSV4-owned parameter and FlexShard policy for Muon."""
    model_config = cast("DeepSeekV4Model.Config", model_spec.model)
    owned = MuonComputeShardingConfig(
        compute_layout=ComputeLayout(shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()})
    )
    attention_shardings = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    expert_projections = ("w1", "w2", "w3")

    def compute_shardings_for_transformer_layer(
        prefix: str,
        layer_config: Any,
        *,
        include_mtp_projections: bool,
    ) -> dict[str, MuonComputeShardingConfig]:
        shardings = {
            f"{prefix}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in attention_shardings.items()
        }
        shardings.update(
            {f"{prefix}.moe.shared_experts.{projection}.weight": owned for projection in expert_projections}
        )
        shardings[f"{prefix}.attention.wq_b.weight"] = MuonComputeShardingConfig(
            compute_layout=ComputeLayout(shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)}),
            compute_view=AttentionPerHeadComputeView(num_heads=layer_config.attention.n_heads),
        )
        shardings[f"{prefix}.attention.wo_a.weight"] = MuonComputeShardingConfig(
            compute_layout=ComputeLayout(shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)}),
            compute_view=AttentionPerHeadComputeView(num_heads=layer_config.attention.n_groups),
        )
        if getattr(layer_config.attention, "compressor", None) is not None:
            for projection in ("wkv", "wgate"):
                shardings[f"{prefix}.attention.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.compressor.ape"] = owned
        expert_sharding = MuonComputeShardingConfig(
            compute_layout=ComputeLayout(
                shardings_by_mesh_axis={
                    MeshAxisName.DP_SHARD.value: Shard(0),
                    MeshAxisName.EFSDP.value: Shard(0),
                    MeshAxisName.EP.value: Shard(0),
                }
            )
        )
        for projection in ("w1_EFD", "w2_EDF", "w3_EFD"):
            shardings[f"{prefix}.moe.routed_experts.inner_experts.{projection}"] = expert_sharding
        shardings[f"{prefix}.moe.router.gate.weight"] = owned
        for module in ("hc_attn_pre", "hc_ffn_pre"):
            shardings[f"{prefix}.{module}.hc_fn"] = owned
        if getattr(layer_config.attention, "indexer", None) is not None:
            for projection in ("wq_b", "weights_proj"):
                shardings[f"{prefix}.attention.indexer.{projection}.weight"] = owned
            for projection in ("wkv", "wgate"):
                shardings[f"{prefix}.attention.indexer.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.indexer.compressor.ape"] = owned
        if include_mtp_projections:
            shardings[f"{prefix}.e_proj.weight"] = owned
            shardings[f"{prefix}.h_proj.weight"] = owned
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
        for layer_id, layer_config in enumerate(model_config.mtp_layers)
    )
    per_layer = main_layer_shardings + mtp_layer_shardings
    compute_sharding_by_fqn = {
        fqn: compute_sharding for layer_shardings in per_layer for fqn, compute_sharding in layer_shardings.items()
    }
    compute_sharding_by_fqn["hc_head.hc_fn"] = owned
    bucket_configs = tuple(
        BucketConfig(name=f"layers.{layer_id}", patterns=tuple(layer_shardings))
        for layer_id, layer_shardings in enumerate(main_layer_shardings)
    )
    bucket_configs += tuple(
        BucketConfig(name=f"mtp_layers.{layer_id}", patterns=tuple(layer_shardings))
        for layer_id, layer_shardings in enumerate(mtp_layer_shardings)
    )
    # hc_head is global rather than layer-scoped, so it needs its own bucket.
    bucket_configs += (BucketConfig(name="hc_head", patterns=("hc_head.hc_fn",)),)
    muon_pattern = (
        r"^(?:"
        r"(?:layers|mtp_layers)\.\d+\.attention\.(?:wq_a|wkv|wo_b|wq_b|wo_a)\.weight"
        r"|(?:layers|mtp_layers)\.\d+\.attention\.indexer\.(?:wq_b|weights_proj)\.weight"
        r"|(?:layers|mtp_layers)\.\d+\.attention\.(?:compressor|indexer\.compressor)\.(?:wkv|wgate)\.weight"
        r"|layers\.\d+\.attention\.(?:compressor|indexer\.compressor)\.ape"
        r"|(?:layers|mtp_layers)\.\d+\.moe\.shared_experts\.w[123]\.weight"
        r"|(?:layers|mtp_layers)\.\d+\.moe\.routed_experts\.inner_experts\.w[123]_[EFD]+"
        r"|(?:layers|mtp_layers)\.\d+\.moe\.router\.gate\.weight"
        r"|(?:layers|mtp_layers)\.\d+\.(?:hc_attn_pre|hc_ffn_pre)\.hc_fn"
        r"|mtp_layers\.\d+\.(?:e_proj|h_proj)\.weight"
        r"|mtp_layers\.\d+\.hc_head\.hc_fn"
        r"|hc_head\.hc_fn"
        r")$"
    )
    return MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistributedMuon": {
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
    """Build a native-default DSV4 optimizer schema with a Muon profile.

    ``name`` remains ``native`` until the user explicitly supplies
    ``--optimizer.name Muon``.  Keeping the profile on the ordinary recipe
    makes the CLI selection the sole Muon entry point.
    """
    native = default_adamw(lr=lr, eps=1e-6)
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
        param_groups=native.param_groups,
        implementation=native.implementation,
        optimizer_factory_kwargs_by_name=native.optimizer_factory_kwargs_by_name,
    )


def _make_trainer_config(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int,
    num_mtp_layers: int = 1,
) -> Trainer.Config:
    model_spec = model_registry(flavor, num_mtp_layers=num_mtp_layers)
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
    return TrainerConfig(
        loss=loss,
        profiler=Profiler.Config(
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
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_debugmodel(*, num_mtp_layers: int = 1) -> Trainer.Config:
    return _make_trainer_config(
        "debugmodel",
        local_batch_size=1,
        seq_len=2048,
        num_mtp_layers=num_mtp_layers,
    )


def deepseek_v4_flash(*, num_mtp_layers: int = 1) -> Trainer.Config:
    return _make_trainer_config(
        "deepseek_v4_flash",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
    )


def deepseek_v4_flash_43layers_16experts(
    *,
    num_mtp_layers: int = 1,
) -> Trainer.Config:
    return _make_trainer_config(
        "deepseek_v4_flash_43layers_16experts",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
    )


def deepseek_v4_pro(*, num_mtp_layers: int = 1) -> Trainer.Config:
    return _make_trainer_config(
        "deepseek_v4_pro",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
    )


def deepseek_v4_pro_61layers_32experts(
    *,
    num_mtp_layers: int = 1,
) -> Trainer.Config:
    return _make_trainer_config(
        "deepseek_v4_pro_61layers_32experts",
        local_batch_size=1,
        seq_len=4096,
        num_mtp_layers=num_mtp_layers,
    )


# --- GraphTrainer config factories ---


def to_graph_trainer_config(
    base_config: Trainer.Config,
    graph_model_registry: Callable[[str], ModelSpec],
) -> GraphTrainer.Config:
    """Project DSV4's NPU recipe config onto GraphTrainer's upstream schema."""
    if isinstance(base_config, TrainerConfig):
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
