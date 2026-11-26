# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 training recipes and shared configuration helpers."""

import os
from typing import TYPE_CHECKING, Literal

import torch
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import CommConfig, OverrideConfig, derive
from torchtitan.config.override import OverrideImport
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.flex_shard import BlockShard, BucketConfig, ComputeLayout
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor

from torchtitan_npu.config import (
    MuonOptimizerProfile,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.config.configs import ProfilerExtensionConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor, compression_alignment
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts
from torchtitan_npu.models.deepseek_v4_1 import EngramArgs, build_model_config
from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model, DeepSeekV41MultimodalModel
from torchtitan_npu.models.deepseek_v4_1.optimizer import EngramTableOptimizerConfig
from torchtitan_npu.models.deepseek_v4_1.trainer import DeepSeekV41Trainer

if TYPE_CHECKING:
    from torchtitan.components.optim.optimizer import BaseOptimizer

    from torchtitan_npu.models.deepseek_v4_1.vision.language_encoder import ReasoningEffort, ThinkingMode


_DEFAULT_HF_ASSETS_PATH = "/path/to/DeepSeekV41_tokenizer"
_DEFAULT_TEXT_DATA_FILES = "tests/assets/c4_test/data.json"
_DEFAULT_SFT_DATASET = "/path/to/train.jsonl"

_A5_PREQUANTIZE_FQNS = [
    ".attention.wq_b",
    ".attention.wkv",
    ".attention.wo_a",
    ".attention.wo_b",
    ".attention.indexer.wq_b",
    ".feed_forward.shared_experts.w2",
]

# The 128-rank full recipe uses this.
_A5_PREQUANTIZE_FQNS_FULL = [
    ".attention.wq_b",
    ".attention.wo_a",
    ".feed_forward.routed_experts.w13",
    ".feed_forward.routed_experts.w2",
]


def _v41_muon_profile(model: DeepSeekV41Model.Config) -> MuonOptimizerProfile:
    """Build the V4.1-owned parameter and FlexShard policy for Muon.

    Same sharding policy as the DSV4 profile where the parameter spaces
    overlap (per-head ``wq_b``/``wo_a`` BlockShard, expert-sharded routed
    GEMMs, Owned for the rest).  V4.1 differences: no MTP depths, no
    ``hc_head``, no compressor APE.  The indexer parameters do receive LI
    gradients on the new baseline, but they stay on the AdamW fallback group
    together with the vision tower and every 1-D parameter — this profile
    does not change the parameter policy the verified stacks ran with.
    """
    model_config = model
    # SimpleFSDP names the dense parameter storage axis fsdp.
    dense_dp_axes = (MeshAxisName.DP_SHARD.value, "fsdp")
    owned = make_owned_layout(dense_dp_axes)
    owned_attention_projections = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_projections = (*owned_attention_projections, "wq_b", "wo_a")
    expert_projections = ("w13", "w2")
    routed_expert_projections = ("w13.weight", "w2.weight")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    expert_sharding = make_expert_layout(dense_dp_axes)

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    bucket_configs: list[BucketConfig] = []
    common_patterns: list[str] = []
    routed_patterns: list[str] = []
    for layer_id, layer_config in enumerate(model_config.layers):
        attention = layer_config.attention
        attention_shardings = {
            f"layers.{layer_id}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in owned_attention_projections.items()
        }
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head and
        # ``wo_a`` one [o_lora_rank, per_group_in] matrix per group, both
        # flattened on dim 0; BlockShard computes Muon per matrix.
        attention_shardings[f"layers.{layer_id}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.head_dim,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        attention_shardings[f"layers.{layer_id}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.wo_a.out_features,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        # Every layer carries a compressor; only the sources own weights.
        for projection in compressor_projections:
            if getattr(attention.compressor, projection, None) is not None:
                attention_shardings[f"layers.{layer_id}.attention.compressor.{projection}.weight"] = owned
        attention_shardings[f"layers.{layer_id}.hc_attn_pre.hc_fn"] = owned
        dense_shardings = {
            f"layers.{layer_id}.feed_forward.shared_experts.{projection}.weight": owned
            for projection in expert_projections
        }
        dense_shardings[f"layers.{layer_id}.feed_forward.router.gate.weight"] = owned
        dense_shardings[f"layers.{layer_id}.hc_ffn_pre.hc_fn"] = owned
        if layer_config.engram is not None:
            # q_weight/k_weight are normalization scales, not projections.
            dense_shardings[f"layers.{layer_id}.engram.gate.wkv"] = owned
        routed_shardings = {
            f"layers.{layer_id}.feed_forward.routed_experts.{projection}": expert_sharding
            for projection in routed_expert_projections
        }
        compute_sharding_by_fqn.update(attention_shardings)
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        common_patterns.extend((*attention_shardings, *dense_shardings))
        routed_patterns.extend(routed_shardings)
        # Pair layers to expose more independent Owned work, but keep routed
        # experts separate to bound redistribution buffers.
        if layer_id % 2 == 1 or layer_id == len(model_config.layers) - 1:
            name = f"layers.{layer_id - layer_id % 2}-{layer_id}"
            bucket_configs.append(BucketConfig(name=f"{name}.common", patterns=tuple(common_patterns)))
            bucket_configs.append(BucketConfig(name=f"{name}.routed", patterns=tuple(routed_patterns)))
            common_patterns.clear()
            routed_patterns.clear()

    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.compressor\.(?:{'|'.join(compressor_projections)})\.weight|"
        rf"feed_forward\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"feed_forward\.routed_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"feed_forward\.router\.gate\.weight|"
        r"engram\.gate\.wkv|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn" + r")$"
    )
    profile = MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=tuple(bucket_configs),
    )
    return with_lora_compute_layouts(model, profile)


def _v41_optimizer_config(model: DeepSeekV41Model.Config, *, include_engram_table: bool = True) -> OptimizerConfig:
    has_engram = any(layer.engram is not None for layer in model.layers)
    profile = _v41_muon_profile(model)
    optimizers: list[BaseOptimizer.Config] = [
        DistMuon.Config(
            pattern=profile.muon_pattern,
            compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
            bucket_configs=profile.bucket_configs,
            lr=1e-5,
            ns_steps=10,
            adjust_lr_fn="match_rms_adamw",
        ),
        AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8, fused=True, foreach=False),
    ]
    # LoRA freezes the host table.  Omitting its sparse group avoids the empty
    # parameter-group error while preserving the Muon/AdamW split for every
    # remaining trainable parameter.  CPT/SFT retain the table group.
    if has_engram and include_engram_table:
        optimizers.insert(
            0,
            EngramTableOptimizerConfig(pattern=r".*\.engram\.table\.weight$", lr=5e-5, betas=(0.9, 0.95), eps=1e-6),
        )
    if has_engram:
        return HostSparseOptimizersContainer.Config(optimizers=optimizers)
    return OptimizerConfig(optimizers=optimizers)


def _vision_dataloader_config(
    model: DeepSeekV41Model.Config,
    *,
    dataset: str = "cc12m-test",
    dataset_path: str | None = None,
    num_packing_bins: int = 0,
):
    """The image-conditioned caption loader, for a multimodal model only.

    Imported here rather than at module scope: the multimodal dataset stack pulls in the
    image-preprocessing dependencies, and a text-only run must not need them.
    """
    from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import caption_dataloader_config

    return caption_dataloader_config(
        dataset=dataset,
        dataset_path=dataset_path,
        num_packing_bins=num_packing_bins,
        per_doc_alignment=compression_alignment(model.compress_ratios),
    )


def _vision_sft_dataloader_config(
    model: DeepSeekV41Model.Config,
    *,
    dataset_path: str = _DEFAULT_SFT_DATASET,
    image_root: str = "",
    thinking_mode: "ThinkingMode" = "chat",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: "ReasoningEffort | None" = None,
) -> GrainDataLoader.Config:
    """The one-chat-context loader graph for a multimodal model.

    This is the whole loader, not a caption-loader variant: the SFT rows carry
    conversations and images, so the caption dataset, packer and collator do not
    participate at all.
    """
    if not isinstance(model, DeepSeekV41MultimodalModel.Config):
        raise ValueError(f"V4.1 vision SFT requires a multimodal flavor, got {type(model).__name__}")

    # Imported here rather than at module scope: the vision dataset stack pulls in
    # the image-preprocessing dependencies a text-only run must not need.
    from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import sft_dataloader_config

    return sft_dataloader_config(
        dataset_path=dataset_path,
        image_root=image_root,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
        per_doc_alignment=compression_alignment(model.compress_ratios),
    )


def _flash_overrides(
    *,
    a5: bool,
    fused_attention: bool,
    moe_comm_backend: Literal["standard", "deepep"] = "standard",
    swap_optimizer: bool = True,
    qat: bool = False,
    vision_rope: bool = True,
) -> list[OverrideImport]:
    """Build the Flash override list for the selected hardware and training path."""
    result: list[OverrideImport] = ["torchtitan_npu.override.common.rms_norm.asc"]
    if vision_rope:
        result.append("torchtitan_npu.override.common.rope.asc_half_rotation")
    result.append("torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post")
    if swap_optimizer:
        result.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    result.append(f"torchtitan_npu.override.common.rope.{'asc_partial' if a5 else 'asc_complex'}")
    if a5:
        result.extend(
            [
                "torchtitan_npu.override.common.swiglu_group.asc",
                "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
            ]
        )
    if fused_attention:
        result.extend(
            [
                "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
                "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
            ]
        )
    if a5 and not qat:
        result.append("torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn")
    dispatcher = "asc_deepep" if moe_comm_backend == "deepep" else "asc"
    result.append(f"torchtitan_npu.override.common.token_dispatcher.{dispatcher}")
    return result


def _make_trainer_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    converters=None,
    parallelism: ParallelismConfig | None = None,
    hf_assets_path: str | None = None,
    moe_comm_backend: str = "standard",
    num_tokens_per_microbatch: int | None = None,
    num_max_tokens_per_rank: int | None = None,
    engram: EngramArgs | None = None,
    engram_enabled: bool = True,
) -> TrainerEx.Config:
    """Assemble the base V4.1 model and trainer configuration."""
    from torchtitan_npu.models.common.moe import dispatcher_capacity

    if seq_len is None:
        seq_len = 512 if "debugmodel" in flavor else 4096
    parallelism = parallelism or ParallelismConfig()
    hf_assets_path = hf_assets_path or os.environ.get("HF_ASSETS_PATH") or _DEFAULT_HF_ASSETS_PATH
    num_tokens = num_tokens_per_microbatch if num_tokens_per_microbatch is not None else seq_len
    if moe_comm_backend == "deepep":
        required = dispatcher_capacity(num_tokens=num_tokens, parallelism=parallelism)
        if num_max_tokens_per_rank is None:
            num_max_tokens_per_rank = required
        elif num_max_tokens_per_rank < required:
            raise ValueError(f"num_max_tokens_per_rank must be at least {required}")
    model = build_model_config(
        flavor,
        converters=converters,
        parallelism=parallelism,
        seq_len=seq_len,
        tokenizer_path=f"{hf_assets_path.rstrip('/')}/tokenizer.json",
        moe_comm_backend=moe_comm_backend,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        engram=engram,
    )
    model = derive(
        model,
        type(model),
        fp32_param_patterns=(
            "layers.*.hc_attn_pre.*",
            "layers.*.hc_ffn_pre.*",
            "layers.*.attention.attn_sink",
        ),
        local_compile_regions=[],
    )
    if not engram_enabled:
        for layer in model.layers:
            layer.engram = None
    if isinstance(model, DeepSeekV41MultimodalModel.Config):
        dataloader = _vision_dataloader_config(model)
    else:
        dataloader = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": _DEFAULT_TEXT_DATA_FILES},
                    ),
                    processor=AlignedTextProcessor.Config(
                        per_doc_alignment=compression_alignment(model.compress_ratios)
                    ),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        )

    return DeepSeekV41Trainer.Config(
        engram_enabled=engram_enabled,
        hf_assets_path=hf_assets_path,
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=dataloader,
        model=model,
        parallelism=parallelism,
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=num_tokens,
            max_context_length=seq_len,
            steps=100,
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        optim=OptimConfig(
            optimizer=_v41_optimizer_config(model),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=20,
                decay_type="cosine",
                min_lr_factor=0.01,
            ),
        ),
        override=OverrideConfig(imports=["torchtitan_npu.override.common.rope.workaround"]),
        activation_remat=FullActivationRematConfig(),
        metrics=MetricsProcessor.Config(log_freq=1),
        comm=CommConfig(init_timeout_seconds=7200, train_timeout_seconds=600),
        checkpointer=None,
        profiler=CANNProfiler.Config(
            enable_profiling=False,
            save_traces_folder="profiling_path",
            extension=ProfilerExtensionConfig(
                enable_online_parse=False, profiler_start=6, profiler_end=7, profile_ranks=[0]
            ),
        ),
    )


def _apply_lora(config: TrainerEx.Config) -> TrainerEx.Config:
    from torchtitan.config.transform import apply_transforms

    from torchtitan_npu.models.common.deepseek_lora import deepseek_v4_lora_transform

    return apply_transforms(config, [deepseek_v4_lora_transform()])


def deepseek_v4_1_debugmodel_multimodal() -> TrainerEx.Config:
    """Build the multimodal debug model for image-caption pretraining."""
    return _make_trainer_config("deepseek_v4_1_debugmodel")


def deepseek_v4_1_debugmodel_multimodal_sft(
    *,
    dataset_path: str = "/path/to/train.jsonl",
    thinking_mode: "ThinkingMode" = "chat",
) -> TrainerEx.Config:
    """Build the multimodal debug SFT recipe with a configurable thinking mode."""
    config = _make_trainer_config("deepseek_v4_1_debugmodel")
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.dataloader = _vision_sft_dataloader_config(
        config.model,
        dataset_path=dataset_path,
        thinking_mode=thinking_mode,
    )
    return config


def deepseek_v4_1_flash_40layers_16experts_text() -> TrainerEx.Config:
    """Build the 16-expert text pretraining recipe for eight A3 ranks at 2K context."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_text",
        seq_len=2048,
        parallelism=ParallelismConfig(data_parallel_shard_degree=8, expert_parallel_degree=8),
    )
    config.override.imports = _flash_overrides(a5=False, fused_attention=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal() -> TrainerEx.Config:
    """Build the 16-expert multimodal pretraining recipe for eight A3 ranks at 2K context."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_vision",
        seq_len=2048,
        parallelism=ParallelismConfig(data_parallel_shard_degree=8, expert_parallel_degree=8),
    )
    config.override.imports = _flash_overrides(a5=False, fused_attention=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft() -> TrainerEx.Config:
    """Build the 16-expert multimodal SFT recipe for eight A3 ranks at 2K context."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_vision",
        seq_len=2048,
        parallelism=ParallelismConfig(data_parallel_shard_degree=8, expert_parallel_degree=8),
    )
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.dataloader = _vision_sft_dataloader_config(config.model)
    config.override.imports = _flash_overrides(a5=False, fused_attention=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_text_lora() -> TrainerEx.Config:
    """Build the 16-expert text LoRA recipe for eight ranks at 2K context."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_text",
        seq_len=2048,
        parallelism=ParallelismConfig(data_parallel_shard_degree=8, expert_parallel_degree=8),
    )
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)])
    config = _apply_lora(config)
    config.checkpointer = DeepSeekV41LoRACheckpointManager.Config(periodic_save_adapter_only=False)
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_bf16() -> TrainerEx.Config:
    """Build the 24-expert multimodal BF16 pretraining recipe for eight A5 ranks."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_24experts_multimodal",
        moe_comm_backend="deepep",
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
            fsdp_reshard_after_forward="always",
        ),
    )
    config.override.imports = _flash_overrides(a5=True, fused_attention=True, moe_comm_backend="deepep")
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_block_fp8() -> TrainerEx.Config:
    """Build the 8-rank multimodal case with the A5 block-FP8 policy."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal_bf16()
    config.training.num_tokens_per_train_step = 64 * config.training.max_context_length
    quantization = config.extension.quantization
    quantization.enable_quantized_training = True
    quantization.recipe = "all_block_fp8"
    quantization.enable_fsdp_prequantize = True
    quantization.fsdp_prequantize_fqns = list(_A5_PREQUANTIZE_FQNS)
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_qat() -> TrainerEx.Config:
    """Build the 8-rank A5 multimodal QAT case with MXFP4 and quantized sparse attention."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal_block_fp8()
    config.override.imports = _flash_overrides(a5=True, fused_attention=True, moe_comm_backend="deepep", qat=True)
    quantization = config.extension.quantization
    quantization.enable_mxfp4_qat = True
    quantization.li_quantization = "mxfp4"
    quantization.enable_sparse_attention_quantization = True
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft() -> TrainerEx.Config:
    """Build the 24-expert multimodal SFT recipe for eight A5 ranks with DeepEP."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_24experts_multimodal",
        moe_comm_backend="deepep",
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
            fsdp_reshard_after_forward="always",
        ),
    )
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.dataloader = _vision_sft_dataloader_config(config.model)
    config.override.imports = _flash_overrides(a5=True, fused_attention=True, moe_comm_backend="deepep")
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_lora() -> TrainerEx.Config:
    """Build the 24-expert multimodal LoRA recipe for eight ranks."""
    config = _make_trainer_config(
        "deepseek_v4_1_flash_40layers_24experts_multimodal",
        seq_len=4096,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
            fsdp_reshard_after_forward="always",
        ),
    )
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)])
    config = _apply_lora(config)
    config.checkpointer = DeepSeekV41LoRACheckpointManager.Config(
        periodic_save_adapter_only=False,
        last_save_in_peft=False,
    )
    return config
