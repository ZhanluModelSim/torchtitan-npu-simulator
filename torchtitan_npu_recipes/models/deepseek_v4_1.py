# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import OverrideConfig, derive
from torchtitan.config.override import OverrideImport
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BlockShard, BucketConfig, ComputeLayout
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import (
    CompileConfig,
    ModelCompileConfig,
    MuonOptimizerProfile,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.config.configs import ProfilerExtensionConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager as NPUCheckpointManager
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor, compression_alignment
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts
from torchtitan_npu.models.deepseek_v4_1 import EngramArgs, model_registry
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model, DeepSeekV41MultimodalModel
from torchtitan_npu.models.deepseek_v4_1.optimizer import EngramTableOptimizerConfig
from torchtitan_npu.models.deepseek_v4_1.trainer import DeepSeekV41Trainer

if TYPE_CHECKING:
    from torchtitan.components.optim.optimizer import BaseOptimizer
    from torchtitan.config.transform import ModelConfigConverter
    from torchtitan.trainer import Trainer

    from torchtitan_npu.models.deepseek_v4_1.vision.language_encoder import ReasoningEffort, ThinkingMode


_A5_PREQUANTIZE_FQNS = [
    ".attention.wq_a",
    ".attention.wq_b",
    ".attention.wkv",
    ".attention.wo_a",
    ".attention.wo_b",
    ".attention.indexer.wq_b",
    ".feed_forward.shared_experts.w13",
    ".feed_forward.shared_experts.w2",
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
    # The target upstream names the dense parameter storage axis dp_shard.
    dense_dp_axes = (MeshAxisName.DP_SHARD.value,)
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
        # Split each layer into three buckets: the routed-expert GEMMs are
        # by far the largest and get a bucket of their own so the
        # Newton-Schulz all-gather peak stays bounded on the 40-layer
        # FSDP8 shape (a single per-layer bucket OOMs at ~89% HBM).
        compute_sharding_by_fqn.update(attention_shardings)
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.attn", patterns=tuple(attention_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.dense", patterns=tuple(dense_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.routed", patterns=tuple(routed_shardings)))

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


def _per_doc_alignment(model: DeepSeekV41Model.Config) -> int:
    """The pooling granularity a packed document must be padded to.

    Derived from the model's compression ratios so the loader and the compressor
    cannot disagree: a document whose length is not a multiple of every ratio would
    get a group straddling its edge.
    """
    alignment = compression_alignment(model.compress_ratios)
    return int(os.environ.get("DSV41_PER_DOC_ALIGNMENT", alignment))


def _vision_dataloader_config(model: DeepSeekV41Model.Config):
    """The image-conditioned caption loader, for a multimodal model only.

    Imported here rather than at module scope: the multimodal dataset stack pulls in the
    image-preprocessing dependencies, and a text-only run must not need them.
    """
    from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import caption_dataloader_config

    return caption_dataloader_config(
        dataset=os.environ.get("DATASET", "cc12m-test"),
        dataset_path=os.environ.get("DATASET_PATH") or None,
        num_packing_bins=int(os.environ.get("PACKING_BUFFER_SIZE", "0")),
        per_doc_alignment=_per_doc_alignment(model),
    )


def _text_data_files() -> str:
    path = os.environ.get("DATA_FILES", "tests/assets/c4_test/data.json")
    if not path.strip():
        raise ValueError("DATA_FILES must be a non-empty JSON/JSONL file path when set")
    return path


def _make_trainer_config(
    flavor: str,
    *,
    converters=None,
    dataloader_factory: Callable[[DeepSeekV41Model.Config], GrainDataLoader.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    moe_comm_backend: str = "standard",
    num_tokens_per_microbatch: int | None = None,
    num_max_tokens_per_rank: int | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    from torchtitan_npu.models.common.moe import dispatcher_capacity

    parallelism = parallelism or ParallelismConfig()
    if parallelism.context_parallel_degree > 1:
        raise NotImplementedError("DeepSeek V4.1 context parallelism requires the v0.4 CP lifecycle migration")
    seq_len = seq_len if seq_len is not None else (512 if "debugmodel" in flavor else 4096)
    hf_assets_path = hf_assets_path or "tests/assets/tokenizer"
    num_tokens = num_tokens_per_microbatch if num_tokens_per_microbatch is not None else seq_len
    if moe_comm_backend == "deepep":
        required = dispatcher_capacity(num_tokens=num_tokens, parallelism=parallelism)
        if num_max_tokens_per_rank is None:
            num_max_tokens_per_rank = required
        elif num_max_tokens_per_rank < required:
            raise ValueError(f"num_max_tokens_per_rank must be at least {required}")
    model = model_registry(
        flavor,
        converters=converters,
        parallelism=parallelism,
        seq_len=seq_len,
        tokenizer_path=os.path.join(hf_assets_path, "tokenizer.json"),
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
    if dataloader_factory is not None:
        dataloader = dataloader_factory(model)
    elif isinstance(model, DeepSeekV41MultimodalModel.Config):
        dataloader = _vision_dataloader_config(model)
    else:
        dataloader = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": _text_data_files()},
                    ),
                    processor=AlignedTextProcessor.Config(per_doc_alignment=_per_doc_alignment(model)),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        )

    return DeepSeekV41Trainer.Config(
        hf_assets_path=hf_assets_path,
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=num_tokens,
            max_context_length=seq_len,
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        model=model,
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=dataloader,
        optim=OptimConfig(optimizer=_v41_optimizer_config(model)),
        parallelism=parallelism,
        activation_remat=None,
        override=OverrideConfig(imports=["torchtitan_npu.override.common.rope.workaround"]),
    )


def _vision_sft_dataloader_config(
    model: DeepSeekV41Model.Config,
    *,
    dataset_path: str,
    image_root: str,
    thinking_mode: "ThinkingMode",
    drop_thinking: bool,
    add_default_bos_token: bool,
    reasoning_effort: "ReasoningEffort | None",
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
        per_doc_alignment=_per_doc_alignment(model),
    )


def deepseek_v4_1_vision_sft(
    *,
    dataset_path: str | None = None,
    image_root: str = "",
    thinking_mode: "ThinkingMode" = "chat",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: "ReasoningEffort | None" = None,
    flavor: str = "deepseek_v4_1_debugmodel",
    converters: list["ModelConfigConverter.Config"] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """A multimodal flavor trained on one chat context per Grain batch.

    The loader is installed while the training config is built, instead of being
    derived from ``_vision_dataloader_config`` afterwards: SFT and caption
    pre-training share the model and trainer policy, but no data graph.
    """
    dataset_path = dataset_path or os.environ.get("DSV41_SFT_DATASET") or os.environ.get("DATASET_PATH", "")
    if not dataset_path:
        raise ValueError("V4.1 vision SFT requires a JSON, JSONL, or Parquet dataset (DSV41_SFT_DATASET)")
    return _make_trainer_config(
        flavor,
        converters=converters,
        dataloader_factory=partial(
            _vision_sft_dataloader_config,
            dataset_path=dataset_path,
            image_root=image_root,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_default_bos_token,
            reasoning_effort=reasoning_effort,
        ),
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def _vision_sft_env(name: str, default: str) -> str:
    """Read one SFT knob for the zero-argument registry entries."""
    return os.environ.get(f"DSV41_SFT_{name}", default)


def _vision_sft_env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(f"DSV41_SFT_{name}")
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _vision_sft_thinking_mode() -> "ThinkingMode":
    mode = _vision_sft_env("THINKING_MODE", "chat")
    if mode not in ("chat", "thinking"):
        raise ValueError(f"DSV41_SFT_THINKING_MODE must be 'chat' or 'thinking', got {mode!r}")
    return mode


def _vision_sft_recipe(
    flavor: str,
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """The ``--config`` entry for SFT: ``--config`` names take no arguments.

    The dataset path falls back to ``DSV41_SFT_DATASET`` / ``DATASET_PATH`` inside
    the recipe; the encoder policy is read here.
    """
    return deepseek_v4_1_vision_sft(
        flavor=flavor,
        image_root=_vision_sft_env("IMAGE_ROOT", ""),
        thinking_mode=_vision_sft_thinking_mode(),
        drop_thinking=_vision_sft_env_flag("DROP_THINKING", True),
        add_default_bos_token=_vision_sft_env_flag("ADD_DEFAULT_BOS_TOKEN", True),
        reasoning_effort=_vision_sft_env("REASONING_EFFORT", "") or None,
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_flash", parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )


def deepseek_v4_1_flash_sft(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _vision_sft_recipe(
        "deepseek_v4_1_flash", parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )


def deepseek_v4_1_flash_text(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_flash_text",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_40layers_16experts_multimodal(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_vision",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _vision_sft_recipe(
        "deepseek_v4_1_flash_40layers_16experts_vision",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_40layers_16experts_text(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_text",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_40layers_24experts_multimodal(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_flash_40layers_24experts_multimodal",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _vision_sft_recipe(
        "deepseek_v4_1_flash_40layers_24experts_multimodal",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_40layers_24experts_text(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_flash_40layers_24experts_text",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_debugmodel_multimodal(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_trainer_config(
        "deepseek_v4_1_debugmodel",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_debugmodel_multimodal_sft(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _vision_sft_recipe(
        "deepseek_v4_1_debugmodel",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_debugmodel_text(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    config = _make_trainer_config(
        "deepseek_v4_1_debugmodel_text",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )
    config.hf_assets_path = hf_assets_path or os.environ.get("HF_ASSETS_PATH") or "tests/assets/deepseek_v3"
    return config


def deepseek_v4_1_stripped(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """The single-card flavor: debug widths, released indexer geometry, ten layers.

    See :func:`torchtitan_npu.models.deepseek_v4_1._stripped` for what is and is not
    shrunk; the point of the flavor is to reach all three branches of the fused selector
    on one card.
    """
    return _make_trainer_config(
        "deepseek_v4_1_stripped", parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )


def deepseek_v4_1_flash_lora(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_lora_trainer_config(
        "deepseek_v4_1_flash", parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )


def deepseek_v4_1_flash_text_lora(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_lora_trainer_config(
        "deepseek_v4_1_flash_text",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def deepseek_v4_1_flash_text_cp(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    raise NotImplementedError("Context parallelism is deferred in the v0.4 eager upgrade.")


def deepseek_v4_1_debugmodel_lora(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    return _make_lora_trainer_config(
        "deepseek_v4_1_debugmodel_text",
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
    )


def _make_lora_trainer_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    from torchtitan.config.transform import apply_transforms

    from torchtitan_npu.models.common.deepseek_lora import deepseek_v4_lora_transform

    config = _make_trainer_config(
        flavor, parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)])
    from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager

    config = apply_transforms(config, [deepseek_v4_lora_transform()])
    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.model, DeepSeekV41Model.Config)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    config.checkpointer = DeepSeekV41LoRACheckpointManager.Config()
    return config


def deepseek_v4_1_flash_cpt(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash(parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram)
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    # The registered converter promotes this upstream schema before CLI parsing.
    cast("Trainer.Config", config).checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash_40layers_16experts_multimodal(
        parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal(
        parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_sft_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_sft(
        parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    cast("Trainer.Config", config).checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_40layers_16experts_multimodal_sft(
        parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft_compiled(
    *,
    parallelism: ParallelismConfig | None = None,
    seq_len: int | None = None,
    hf_assets_path: str | None = None,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal_sft(
        parallelism=parallelism, seq_len=seq_len, hf_assets_path=hf_assets_path, engram=engram
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def _reject_legacy_overrides() -> None:
    for name in ("CLI_OVERRIDES", "OPTIMIZER_OVERRIDES", "DATALOADER_OVERRIDES", "MHC_POST_OVERRIDE"):
        if name in os.environ:
            raise ValueError(f"{name} no longer selects training settings; define a Python recipe or use --override")


def _world_size(*, local_processes: int, nodes: int) -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    node_ips = os.environ.get("NODE_IPS", "")
    node_count = len([address for address in node_ips.split(",") if address.strip()]) if node_ips else nodes
    return int(os.environ.get("NGPU", local_processes)) * int(os.environ.get("NNODES", node_count))


def _topology(
    *,
    local_processes: int,
    nodes: int,
    shard_degree: int,
    expert_degree: int,
    context_degree: int = 1,
    configurable: bool = False,
) -> ParallelismConfig:
    if configurable:
        shard_degree = int(os.environ.get("DP_SHARD", shard_degree))
        expert_degree = int(os.environ.get("EP", expert_degree))
        context_degree = int(os.environ.get("CP", context_degree))
    world_size = _world_size(local_processes=local_processes, nodes=nodes)
    if (
        min(world_size, shard_degree, expert_degree, context_degree) <= 0
        or world_size % (shard_degree * context_degree)
        or world_size % expert_degree
    ):
        raise ValueError("WORLD_SIZE must form complete DP/CP/EP groups, and EP must divide the model's expert count")
    return ParallelismConfig(
        data_parallel_shard_degree=shard_degree,
        data_parallel_replicate_degree=world_size // (shard_degree * context_degree),
        expert_parallel_degree=expert_degree,
        context_parallel_degree=context_degree,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward="always",
        enable_sequence_parallel=True,
    )


def _set_context(config: TrainerEx.Config, *, sequence_length: int, token_budget: int, steps: int) -> None:
    assert isinstance(config.model, DeepSeekV41Model.Config)
    if config.model.max_context_length != sequence_length:
        raise ValueError("Construct the model with the training sequence length")
    config.training.num_tokens_per_microbatch_per_dp_rank = sequence_length
    config.training.num_tokens_per_train_step = token_budget
    config.training.steps = steps
    config.training.disable_cuda_graphs = True
    config.metrics.log_freq = 1


def _set_muon(config: TrainerEx.Config, *, include_engram_table: bool = True) -> None:
    assert isinstance(config.model, DeepSeekV41Model.Config)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    config.optim.optimizer = _v41_optimizer_config(config.model, include_engram_table=include_engram_table)


def _set_schedule(config: TrainerEx.Config) -> None:
    config.optim.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=2,
        total_steps=config.training.steps,
        decay_ratio=1.0,
        decay_type="cosine",
        min_lr_factor=0.01,
    )


def _overrides(*, a5: bool, debug: bool, swap: bool = True, qat: bool = False) -> list[OverrideImport]:
    golden = os.environ.get("USE_GOLDEN", "0") == "1"
    tilelang = a5 and not golden and os.environ.get("USE_TILELANG", "0") == "1"
    result: list[OverrideImport]
    if golden:
        result = ["torchtitan_npu.override.common.rope.workaround"]
    else:
        result = [
            "torchtitan_npu.override.common.rms_norm.asc",
            "torchtitan_npu.override.common.rope.asc_half_rotation",
            (
                "torchtitan_npu.override.deepseek_v4_1.tilelang.tilelang_hc_post"
                if tilelang
                else "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post"
            ),
            "torchtitan_npu.override.common.token_dispatcher.asc",
        ]
    if swap:
        result.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    # QAT historically selects its fused attention list independently of USE_GOLDEN.
    if qat or not golden:
        result.append(f"torchtitan_npu.override.common.rope.{'asc_partial' if a5 else 'asc_complex'}")
        if tilelang and not qat:
            result.append("torchtitan_npu.override.deepseek_v4_1.tilelang.tilelang_hc_pre")
        if a5:
            result.extend(
                [
                    "torchtitan_npu.override.common.swiglu_group.asc",
                    "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
                ]
            )
        if a5 or debug:
            result.extend(
                [
                    "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
                    "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
                ]
            )
        if a5 and not tilelang and not qat:
            result.append("torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn")
    return result


def _sft(
    flavor: str,
    *,
    checkpoint: bool,
    parallelism: ParallelismConfig,
    seq_len: int,
    hf_assets_path: str,
    engram: EngramArgs | None = None,
) -> TrainerEx.Config:
    config = deepseek_v4_1_vision_sft(
        flavor=flavor,
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        engram=engram,
        dataset_path=os.environ.get("DSV41_SFT_DATASET") or os.environ.get("DATASET_PATH") or "/path/to/train.jsonl",
        image_root=_vision_sft_env("IMAGE_ROOT", ""),
        thinking_mode=_vision_sft_thinking_mode(),
        drop_thinking=_vision_sft_env_flag("DROP_THINKING", True),
        add_default_bos_token=_vision_sft_env_flag("ADD_DEFAULT_BOS_TOKEN", True),
        reasoning_effort=_vision_sft_env("REASONING_EFFORT", "") or None,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    if checkpoint:
        config.checkpointer = NPUCheckpointManager.Config(load_only=True)
    return config


def _full(*, a5: bool, sft: bool = False, lora: bool = False) -> TrainerEx.Config:
    _reject_legacy_overrides()
    parallelism = _topology(
        local_processes=8 if a5 else 16,
        nodes=16 if a5 else 8,
        shard_degree=128,
        expert_degree=128,
        configurable=True,
    )
    options = dict(
        parallelism=parallelism,
        seq_len=4096,
        hf_assets_path=os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer"),
    )
    if sft:
        config = _sft("deepseek_v4_1_flash", checkpoint=True, **options)
    elif lora:
        config = deepseek_v4_1_flash_lora(**options)
    else:
        config = deepseek_v4_1_flash_cpt(**options)
    _set_context(config, sequence_length=4096, token_budget=4194304, steps=int(os.environ.get("STEPS", "100")))
    _set_muon(config, include_engram_table=not lora)
    _set_schedule(config)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer")
    config.debug.print_config = True
    config.profiler = CANNProfiler.Config(
        enable_profiling=False,
        save_traces_folder="profiling_path",
        extension=ProfilerExtensionConfig(
            enable_online_parse=False, profiler_start=6, profiler_end=7, profile_ranks=[0]
        ),
    )
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    assert config.checkpointer is not None
    config.checkpointer.folder = os.environ.get("CKPT_SAVE_LOAD_PATH", "/path/to/save_ckpt")
    config.checkpointer.initial_load_path = os.environ.get("CKPT_INIT_LOAD_PATH", "/path/to/init_load_ckpt")
    config.checkpointer.initial_load_in_hf = True
    config.checkpointer.load_only = not lora
    config.override.imports = _overrides(a5=a5, debug=False, swap=not lora)
    config.extension.quantization.enable_quantized_training = a5
    config.extension.quantization.recipe = "all_block_fp8"
    if a5:
        config.extension.quantization.enable_fsdp_prequantize = True
        config.extension.quantization.fsdp_prequantize_fqns = list(_A5_PREQUANTIZE_FQNS)
    return config


def deepseek_v4_1_flash_cpt_4k_a3() -> TrainerEx.Config:
    return _full(a5=False)


def deepseek_v4_1_flash_cpt_4k_a5() -> TrainerEx.Config:
    return _full(a5=True)


def deepseek_v4_1_flash_lora_4k_a3() -> TrainerEx.Config:
    return _full(a5=False, lora=True)


def deepseek_v4_1_flash_sft_4k_a3() -> TrainerEx.Config:
    return _full(a5=False, sft=True)


def deepseek_v4_1_flash_sft_4k_a5() -> TrainerEx.Config:
    return _full(a5=True, sft=True)


def deepseek_v4_1_flash_cpt_1m_a5() -> TrainerEx.Config:
    """Declare the long-context CP configuration; the model still rejects unsupported CP."""
    _reject_legacy_overrides()
    parallelism = _topology(
        local_processes=8,
        nodes=16,
        shard_degree=1,
        expert_degree=128,
        context_degree=128,
        configurable=True,
    )
    config = deepseek_v4_1_flash_text(
        parallelism=parallelism,
        seq_len=1048576,
        hf_assets_path=os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer"),
    )
    config = derive(config, type(config), engram_enabled=False)
    _set_context(config, sequence_length=1048576, token_budget=1048576, steps=int(os.environ.get("STEPS", "10")))
    _set_muon(config)
    _set_schedule(config)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer")
    config.debug.print_config = True
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    packed = config.dataloader.dataset
    assert isinstance(packed, ConcatThenSplitPackingConfig)
    dataset = packed.dataset
    assert isinstance(dataset, SingleDatasetConfig)
    assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
    assert isinstance(dataset.processor, AlignedTextProcessor.Config)
    dataset.source.load_dataset_kwargs["data_files"] = (
        os.environ.get("DATA_FILES") or os.environ.get("DATASET_PATH") or "tests/assets/c4_test/data.json"
    )
    dataset.processor.per_doc_alignment = int(os.environ.get("DSV41_PER_DOC_ALIGNMENT", "256"))
    config.profiler = CANNProfiler.Config(
        enable_profiling=False,
        save_traces_folder="profiling_path",
        extension=ProfilerExtensionConfig(
            enable_online_parse=False, profiler_start=6, profiler_end=7, profile_ranks=[0]
        ),
    )
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.common.rope.asc_partial",
        "torchtitan_npu.override.common.token_dispatcher.asc",
        "torchtitan_npu.override.common.swiglu_group.asc",
        "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
        "torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn",
        "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post",
        "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
    ]
    return config
