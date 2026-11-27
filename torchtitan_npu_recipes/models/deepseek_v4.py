# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full Flash CPT, SFT, LoRA and QAT recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig, FullActivationRematConfig
from torchtitan.distributed.flex_shard import BlockShard, BucketConfig, ComputeLayout
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.observability.metrics import MetricsProcessor

from torchtitan_npu.config import (
    MuonOptimizerProfile,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.config.configs import KVNormQuantizationConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts
from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu.models.deepseek_v4.chat_processor import DeepSeekV4ChatProcessor
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper
from torchtitan_npu.override.deepseek_v4.cp_metadata import document_alignment

if TYPE_CHECKING:
    from torchtitan.config.transform import ModelConfigConverter


def _trainer_config(
    model: DeepSeekV4Model.Config,
    *,
    parallelism: ParallelismConfig,
    num_mtp_layers: int = 0,
    hf_assets_path: str | None = None,
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int | None = None,
) -> TrainerEx.Config:
    """Assemble the shared text-training components around an explicit model."""
    if parallelism.context_parallel_degree > 1:
        from torchtitan_npu.override.deepseek_v4.context_parallel import configure_context_parallel

        model = configure_context_parallel(model, parallelism)
    model.local_compile_regions = []
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
                        path="json", split="train", load_dataset_kwargs={"data_files": data_files}
                    ),
                    processor=AlignedTextProcessor.Config(
                        per_doc_alignment=document_alignment(model.compress_ratios, parallelism)
                        if per_doc_alignment is None
                        else per_doc_alignment
                    ),
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
            num_tokens_per_microbatch_per_dp_rank=model.max_context_length,
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


def _flash_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int = 4096,
    num_mtp_layers: int = 0,
    num_layers: int | None = None,
    hf_assets_path: str | None = None,
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> TrainerEx.Config:
    model = build_model_config(
        flavor, parallelism=parallelism, seq_len=seq_len, num_mtp_layers=num_mtp_layers, converters=converters
    )
    if num_layers is not None:
        # Muon derives its layout and buckets from the final main and MTP layers.
        model.layers = model.layers[:num_layers]
        model.n_layers = len(model.layers)
        model.compress_ratios = model.compress_ratios[:num_layers]
    return _trainer_config(
        model,
        parallelism=parallelism,
        num_mtp_layers=num_mtp_layers,
        hf_assets_path=hf_assets_path,
        data_files=data_files,
        per_doc_alignment=per_doc_alignment,
    )


# Pre-quantize whitelist for the 8-rank eager sample: dim-0 projections whose
# local shard keeps a multiple of 64 rows at ``dp_shard=8``.  ``.attention.wkv``
# is 512 rows, so 16-way sharding would trip the wrapper's axis=-2 block guard.
# ``.attention.wq_a`` (``q_lora_rank`` 1024 rows -> 128 per rank) is V4-only;
# V4.1's 1280 rows are 160 per rank and stay out.  The 128-rank full recipe uses
# ``_A5_PREQUANTIZE_FQNS_FULL`` instead.
_A5_PREQUANTIZE_FQNS = [
    ".attention.wq_a",
    ".attention.wq_b",
    ".attention.wkv",
    ".attention.wo_a",
    ".attention.wo_b",
    ".attention.indexer.wq_b",
    ".feed_forward.shared_experts.w2",
]

# 8-rank entries that leave the full list: rows per rank at ``dp_shard=128``,
# none of them on the 64-row block grid:
#   .attention.wkv                     512 ->  4
#   .attention.wq_a                   1024 ->  8
#   .attention.wo_b                   4096 -> 32
#   .feed_forward.shared_experts.w2   4096 -> 32
# Kept: ``.attention.wq_b`` 32768 -> 256, ``.attention.wo_a`` 8192 -> 64,
# ``.attention.indexer.wq_b`` 8192 -> 64.
# Routed experts stay whitelisted: FSDP keeps them on the expert axis (``w13``
# ``[E, 2, F, D]``, ``w2`` ``[E, dim, F]``) while the expert-axis shard size
# ``edp_shard * ep`` (= ``dp_shard * cp * tp``) stays within ``num_experts``, so
# ``F``/``dim`` are never split.  ``ep == dp_shard`` leaves ``edp_shard`` 1: a
# structural no-op that only engages once ``ep < dp_shard``.
_A5_PREQUANTIZE_FQNS_FULL = [
    ".attention.wq_b",
    ".attention.wo_a",
    ".attention.indexer.wq_b",
    ".feed_forward.routed_experts.w13",
    ".feed_forward.routed_experts.w2",
]


def _enable_indexer_aux_loss(model: DeepSeekV4Model.Config) -> None:
    """Train the reference selector through its existing distillation loss."""
    for layer in model.layers:
        if layer.attention.indexer is not None:
            inner = layer.attention.inner_attention
            inner.aux_loss = SparseIndexerLoss.Config(
                coeff=1.0,
                reduce_mesh="loss",
                mass_weighted=True,
                softmax_scale=inner.softmax_scale,
                num_heads=layer.attention.n_heads,
                compress_ratio=inner.compress_ratio,
            )


def _dsv4_muon_profile(model: DeepSeekV4Model.Config) -> MuonOptimizerProfile:
    """Build the DSV4-owned parameter and FlexShard policy for Muon."""
    model_config = model
    # GraphTrainer SimpleFSDP flattens the dense DP mesh to "fsdp".
    dense_dp_axes = (MeshAxisName.DP_SHARD.value, "fsdp")
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
                axis: BlockShard(dim=0, block_sizes=(attention.o_lora_rank,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        for compressor_name in ("compressor", "compressor_128"):
            if getattr(attention, compressor_name, None) is not None:
                for projection in compressor_projections:
                    shardings[f"{prefix}.attention.{compressor_name}.{projection}.weight"] = owned
                shardings[f"{prefix}.attention.{compressor_name}.ape"] = owned
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
    compute_sharding_by_fqn = {}
    for layer_shardings in per_layer:
        compute_sharding_by_fqn.update(layer_shardings)
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
        rf"attention\.(?:compressor(?:_128)?|indexer\.compressor)\.(?:{'|'.join(compressor_projections)})\.weight|"
        r"attention\.(?:compressor(?:_128)?|indexer\.compressor)\.ape|"
        rf"feed_forward\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"feed_forward\.routed_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"feed_forward\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn|"
        rf"(?:{'|'.join(mtp_projections)})\.weight|"
        r"hc_head\.hc_fn" + r")$"
    )
    profile = MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
    )
    return with_lora_compute_layouts(model, profile)


def _dsv4_optimizer_config(model: DeepSeekV4Model.Config, *, lr: float) -> OptimizerConfig:
    """Build the recipe-owned Muon + AdamW groups for DSV4."""
    profile = _dsv4_muon_profile(model)
    return OptimizerConfig(
        optimizers=[
            DistMuon.Config(
                pattern=profile.muon_pattern,
                compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
                bucket_configs=profile.bucket_configs,
                lr=lr,
                momentum=0.95,
                nesterov=True,
                ns_steps=10,
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(
                pattern=r".*",
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                fused=True,
                foreach=False,
            ),
        ]
    )


def deepseek_v4_flash_cpt_4k_a3() -> TrainerEx.Config:
    """Build 128-rank Flash CPT on A3 (8 x 16 NPUs) with BF16, standard MoE communication and fused ops."""
    parallelism = ParallelismConfig(
        expert_parallel_degree=128,
        data_parallel_shard_degree=128,
        data_parallel_replicate_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        fsdp_reshard_after_forward="always",
    )
    model = build_model_config("deepseek_v4_flash", parallelism=parallelism, seq_len=4096)
    model.local_compile_regions = []
    optimizer = _dsv4_optimizer_config(model, lr=1e-5)
    config = TrainerEx.Config(
        hf_assets_path="/path/to/DeepSeekV4_tokenizer",
        model=model,
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model))),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json", split="train", load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"}
                    ),
                    processor=AlignedTextProcessor.Config(per_doc_alignment=128),
                    post_filters=(lambda sample: sample is not None,),
                )
            ),
        ),
        optim=OptimConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=20, decay_ratio=0.8, decay_type="cosine", min_lr_factor=0.01
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=4096,
            num_tokens_per_train_step=4194304,
            max_context_length=4096,
            steps=100,
        ),
        parallelism=parallelism,
        activation_remat=FullActivationRematConfig(),
        compile=None,
        checkpointer=CheckpointManager.Config(
            load_only=True,
            folder="/path/to/save_ckpt",
            initial_load_path="/path/to/init_load_ckpt",
            initial_load_in_hf=True,
            interval=100,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        profiler=CANNProfiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
            save_traces_folder="profiling_path",
        ),
    )
    config.debug.print_config = True
    config.profiler.extension.enable_online_parse = False
    config.profiler.extension.profiler_start = 6
    config.profiler.extension.profiler_end = 7
    config.profiler.extension.profile_ranks = [0]
    config.comm.init_timeout_seconds = 7200
    config.comm.train_timeout_seconds = 600
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
        "torchtitan_npu.override.common.rope.asc_complex",
        "torchtitan_npu.override.common.token_dispatcher.asc",
    ]
    return config


def deepseek_v4_flash_cpt_4k_a5() -> TrainerEx.Config:
    """Build 128-rank Flash CPT on A5 (16 x 8 NPUs) with DeepEP and block-FP8 training."""
    config = deepseek_v4_flash_cpt_4k_a3()
    from torchtitan.models.common.moe import RoutedExperts

    from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher

    # [TODO] The v0.4.0 model has no backend field; select the dispatcher config itself.
    assert config.model is not None
    for _, experts, _, _ in config.model.traverse(RoutedExperts.Config):
        dispatcher = experts.token_dispatcher
        experts.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=dispatcher.num_experts,
            top_k=dispatcher.top_k,
            hidden_dim=experts.w13.in_features,
            num_max_tokens_per_rank=8192,
            absorb_router_scores=True,
        )
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.activation_remat = DefaultActivationRematConfig()
    quantization = config.extension.quantization
    quantization.enable_quantized_training = True
    quantization.recipe = "all_block_fp8"
    quantization.enable_fsdp_prequantize = True
    quantization.fsdp_prequantize_fqns = list(_A5_PREQUANTIZE_FQNS_FULL)
    quantization.li_quantization = "fp8"
    quantization.kv_norm_quantization = KVNormQuantizationConfig(
        format="mxfp8",
        fqns=[".attention.kv_norm", ".attention.compressor.norm", ".attention.compressor_128.norm"],
        block_size=64,
    )
    config.override.imports = [
        "torchtitan_npu.override.deepseek_v4.common.rms_norm",
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.deepseek_v4.common.rope",
        "torchtitan_npu.override.deepseek_v4.compressor.asc",
        "torchtitan_npu.override.common.swiglu_group.asc",
        "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
        "torchtitan_npu.override.common.token_dispatcher.asc_deepep",
    ]
    return config


def deepseek_v4_flash_cpt_1024k_a5() -> TrainerEx.Config:
    """BF16 Flash CPT with document-aligned packing on 128 context ranks."""
    from dataclasses import replace

    from torchtitan.config import derive
    from torchtitan.models.common import RoPE

    from torchtitan_npu.config.configs import QuantizationExtensionConfig
    from torchtitan_npu.override.deepseek_v4.context_parallel import configure_context_parallel

    config = deepseek_v4_flash_cpt_4k_a5()
    config.extension.quantization = QuantizationExtensionConfig()
    parallelism = replace(
        config.parallelism,
        context_parallel_degree=128,
        data_parallel_shard_degree=1,
        context_parallel_load_balancer=None,
    )
    length = 1_048_576
    model = config.model
    assert isinstance(model, DeepSeekV4Model.Config)
    model.max_context_length = length
    model.local_compile_regions = []
    for _, rope, _, _ in model.traverse(RoPE.Config):
        rope.max_context_length = length
    config.training.max_context_length = length
    config.training.num_tokens_per_microbatch_per_dp_rank = length
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    packed = config.dataloader.dataset
    assert isinstance(packed, ConcatThenSplitPackingConfig) and isinstance(packed.dataset, SingleDatasetConfig)
    assert isinstance(packed.dataset.processor, AlignedTextProcessor.Config)
    model = configure_context_parallel(model, parallelism)
    packed.dataset.processor.per_doc_alignment = document_alignment(model.compress_ratios, parallelism)
    return derive(config, type(config), parallelism=parallelism, model=model, compile=None)


def deepseek_v4_flash_qat_4k_a5() -> TrainerEx.Config:
    """Build 128-rank Flash QAT on A5 (16 x 8 NPUs) with DeepEP, block-FP8 training and MXFP4 QAT."""
    config = deepseek_v4_flash_cpt_4k_a5()
    config.extension.quantization.enable_mxfp4_qat = True
    config.extension.quantization.li_quantization = "mxfp4"
    return config


def deepseek_v4_flash_sft_4k_a3() -> TrainerEx.Config:
    """Build 128-rank Flash SFT on A3 (8 x 16 NPUs) with BF16, chat data and checkpoint saving."""
    config = deepseek_v4_flash_cpt_4k_a3()
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json", split="train", load_dataset_kwargs={"data_files": "/path/to/train.jsonl"}
                ),
                processor=DeepSeekV4ChatProcessor.Config(per_doc_alignment=128),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=1,
        ),
    )
    assert config.checkpointer is not None
    config.checkpointer.load_only = False
    return config


def deepseek_v4_flash_lora_4k_a3() -> TrainerEx.Config:
    """Build 128-rank Flash LoRA on A3 (8 x 16 NPUs) with BF16, rank 16 adapters and PEFT checkpoints."""
    from torchtitan.config.transform import apply_transforms

    from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform
    from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager

    config = deepseek_v4_flash_cpt_4k_a3()
    config = apply_transforms(config, [deepseek_v4_lora_transform(rank=16, alpha=32)])
    assert isinstance(config.model, DeepSeekV4Model.Config)
    config.optim.optimizer = _dsv4_optimizer_config(config.model, lr=1e-5)
    config.checkpointer = DeepSeekV4PEFTCheckpointManager.Config(
        load_only=False,
        save_training_state=True,
        interval=100,
        folder="/path/to/save_ckpt",
        initial_load_path="/path/to/init_load_ckpt",
        initial_load_in_hf=True,
    )
    return config
