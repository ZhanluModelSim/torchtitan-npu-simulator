# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, partial
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
from torchtitan.components.optim import AdamW, BaseOptimizer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import (
    CompileConfig,
    ModelCompileConfig,
    MuonOptimizerProfile,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
    HostSparseOptim,
    HostSparseOptimizersContainer,
    SparseAdam,
)
from torchtitan_npu.extensions.torchtitan.trainer import NPUTrainingEngine, TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor, compression_alignment
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts

from . import model_registry
from .indexer import Selector
from .model import DeepSeekV41Model, DeepSeekV41MultimodalModel

if TYPE_CHECKING:
    from torchtitan.config.transform import ModelConfigConverter
    from torchtitan.trainer import Trainer

    from .vision.language_encoder import ReasoningEffort, ThinkingMode


def _per_doc_alignment(model: DeepSeekV41Model.Config) -> int:
    """The pooling granularity a packed document must be padded to.

    Derived from the model's compression ratios so the loader and the compressor
    cannot disagree: a document whose length is not a multiple of every ratio would
    get a group straddling its edge.
    """
    alignment = compression_alignment(model.compress_ratios)
    return int(os.environ.get("DSV41_PER_DOC_ALIGNMENT", alignment))


@dataclass(kw_only=True)
class EngramTableOptimizerConfig(SparseAdam.Config):
    """The recipe-owned table group, distinct from user-defined groups."""


class DeepSeekV41TrainingEngine(NPUTrainingEngine):
    """Set the LI objective denominator before model preprocessing shards inputs."""

    @cached_property
    def _li_selectors(self) -> list[Selector]:
        # Resolve after model initialization has applied overrides and wrappers.
        return [
            module
            for part in self.model_parts
            for module in part.modules()
            if isinstance(module, Selector) and type(module) is not Selector
        ]

    def forward_backward(self, *, microbatch_groups, global_valid_tokens):
        if self._li_selectors:
            num_queries = sum(
                microbatch.as_input_dict()["input"].numel() for group in microbatch_groups for microbatch in group
            )
            denominator = torch.tensor(num_queries, device=self.device, dtype=torch.float32)
            if self.parallelism_context.dp_enabled:
                denominator = dist_utils.dist_sum_tensor(denominator, self.parallelism_context.get_mesh("dp"))
            for selector in self._li_selectors:
                selector.num_global_queries = denominator
        return super().forward_backward(microbatch_groups=microbatch_groups, global_valid_tokens=global_valid_tokens)


class DeepSeekV41Trainer(TrainerEx):
    """V4.1 trainer with an engine-owned LI objective denominator."""

    engine_cls = DeepSeekV41TrainingEngine

    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        """Expose the model switch while DeepSeekV41Model.Config is suppressed from the CLI."""

        engram_enabled: bool = True

        def __post_init__(self) -> None:
            assert self.model is not None and isinstance(self.model, DeepSeekV41Model.Config)
            if not self.engram_enabled:
                model = copy.deepcopy(self.model)
                self.model = model
                assert isinstance(model, DeepSeekV41Model.Config)
                # Remove the HostSparse capability together with Engram.
                optimizer = derive(copy.deepcopy(self.optim.optimizer), OptimizerConfig)
                self.optim.optimizer = optimizer
                for layer in model.layers:
                    layer.engram = None
                optimizer.optimizers = [
                    group for group in optimizer.optimizers if not isinstance(group, EngramTableOptimizerConfig)
                ]
                optimizer._optimizer_defaults = {
                    name: [group for group in groups if not isinstance(group, EngramTableOptimizerConfig)]
                    for name, groups in optimizer._optimizer_defaults.items()
                }
            TrainerEx.Config.__post_init__(self)


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
        )
        attention_shardings[f"layers.{layer_id}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.wo_a.out_features,)) for axis in dense_dp_axes
            },
        )
        # Every layer carries a compressor; only the sources own weights.
        for projection in compressor_projections:
            if getattr(attention.compressor, projection, None) is not None:
                attention_shardings[f"layers.{layer_id}.attention.compressor.{projection}.weight"] = owned
        attention_shardings[f"layers.{layer_id}.hc_attn_pre.hc_fn"] = owned
        dense_shardings = {
            f"layers.{layer_id}.moe.shared_experts.{projection}.weight": owned for projection in expert_projections
        }
        dense_shardings[f"layers.{layer_id}.moe.router.gate.weight"] = owned
        dense_shardings[f"layers.{layer_id}.hc_ffn_pre.hc_fn"] = owned
        if layer_config.engram is not None:
            # q_weight/k_weight are normalization scales, not projections.
            dense_shardings[f"layers.{layer_id}.engram.gate.wkv"] = owned
        routed_shardings = {
            f"layers.{layer_id}.moe.routed_experts.{projection}": expert_sharding
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
        rf"moe\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"moe\.routed_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"moe\.router\.gate\.weight|"
        r"engram\.gate\.wkv|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn" + r")$"
    )
    from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon

    profile = MuonOptimizerProfile(
        optimizer_config_type=DistMuon.Config,
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": tuple(bucket_configs),
            }
        },
    )
    return with_lora_compute_layouts(model, profile)


def _v41_optimizer_config(model: DeepSeekV41Model.Config) -> OptimizerConfig:
    has_engram = any(layer.engram is not None for layer in model.layers)
    optimizer_type = HostSparseOptimizersContainer.Config if has_engram else OptimizerConfig
    profile = _v41_muon_profile(model)
    optimizers: list[BaseOptimizer.Config] = [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]
    if has_engram:
        optimizers.insert(
            0,
            EngramTableOptimizerConfig(
                pattern=r".*\.engram\.table\.weight$",
                lr=5e-5,
                betas=(0.9, 0.95),
                eps=1e-6,
            ),
        )
    return optimizer_type(
        name="AdamW" if has_engram else "Muon",
        _optimizer_defaults={
            "AdamW": optimizers,
            "Muon": [
                profile.optimizer_config_type(
                    pattern=profile.muon_pattern,
                    lr=1e-5,
                    adjust_lr_fn="match_rms_adamw",
                    **profile.optimizer_factory_kwargs["DistMuon"],
                ),
                AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8, fused=True, foreach=False),
            ],
        },
        _muon_profile=profile,
    )


def _vision_dataloader_config(model: DeepSeekV41Model.Config):
    """The image-conditioned caption loader, for a multimodal model only.

    Imported here rather than at module scope: the multimodal dataset stack pulls in the
    image-preprocessing dependencies, and a text-only run must not need them.
    """
    from .vision.dataloader import caption_dataloader_config

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
) -> TrainerEx.Config:
    model = model_registry(flavor, converters=converters)
    if model.n_layers != len(model.layers):  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not describe every configured layer")

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

    from torchtitan_npu.patches.torchtitan.distributed.context_parallel import DocChunkCPLoadBalancer

    return DeepSeekV41Trainer.Config(
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=min(4096, model.max_context_length),
            max_context_length=min(4096, model.max_context_length),
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        model=model,
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=dataloader,
        optim=HostSparseOptim.Config(optimizer=_v41_optimizer_config(model)),
        parallelism=ParallelismConfig(context_parallel_load_balancer=DocChunkCPLoadBalancer.Config()),
        activation_checkpoint=FullAC.Config(),
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
    from .vision.language_dataset import sft_dataloader_config

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


def _vision_sft_recipe(flavor: str) -> TrainerEx.Config:
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
    )


def deepseek_v4_1_flash() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash")


def deepseek_v4_1_flash_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_flash")


def deepseek_v4_1_flash_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_text")


def deepseek_v4_1_flash_40layers_16experts_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_16experts_vision")


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_flash_40layers_16experts_vision")


def deepseek_v4_1_flash_40layers_16experts_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_16experts_text")


def deepseek_v4_1_flash_40layers_24experts_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_24experts_multimodal")


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_flash_40layers_24experts_multimodal")


def deepseek_v4_1_flash_40layers_24experts_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_24experts_text")


def deepseek_v4_1_debugmodel_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_debugmodel")


def deepseek_v4_1_debugmodel_multimodal_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_debugmodel")


def deepseek_v4_1_debugmodel_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_debugmodel_text")


def deepseek_v4_1_stripped() -> TrainerEx.Config:
    """The single-card flavor: debug widths, released indexer geometry, ten layers.

    See :func:`torchtitan_npu.models.deepseek_v4_1._stripped` for what is and is not
    shrunk; the point of the flavor is to reach all three branches of the fused selector
    on one card.
    """
    return _make_trainer_config("deepseek_v4_1_stripped")


def deepseek_v4_1_flash_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_flash")


def deepseek_v4_1_flash_text_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_flash_text")


def deepseek_v4_1_flash_text_cp() -> TrainerEx.Config:
    """Text Flash with the fused document-chunk CP backend selected explicitly."""
    from torchtitan.config.transform import ContextParallelTransform

    from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41ContextParallelSparseAttention

    config = deepseek_v4_1_flash_text()
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.model = cast(
        "DeepSeekV41Model.Config",
        ContextParallelTransform(inner_attention=AscV41ContextParallelSparseAttention).transform(config.model),
    )
    config.parallelism.context_parallel_degree = 2
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, ConcatThenSplitPackingConfig)
    assert isinstance(config.dataloader.dataset.dataset, SingleDatasetConfig)
    processor = config.dataloader.dataset.dataset.processor
    assert isinstance(processor, AlignedTextProcessor.Config)
    processor.per_doc_alignment = max(
        processor.per_doc_alignment, compression_alignment(config.model.compress_ratios) * 2
    )
    config.override.imports.append("torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc")
    config.__post_init__()
    return config


def deepseek_v4_1_debugmodel_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_debugmodel_text")


def _make_lora_trainer_config(flavor: str) -> TrainerEx.Config:
    from torchtitan.config.transform import apply_transforms

    from torchtitan_npu.models.common.deepseek_lora import deepseek_v4_lora_transform

    config = _make_trainer_config(flavor)
    config.optim.optimizer = OptimizerConfig(
        _optimizer_defaults={"AdamW": [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]},
    )
    from .checkpoint import DeepSeekV41LoRACheckpointManager

    config = apply_transforms(config, [deepseek_v4_lora_transform()])
    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.model, DeepSeekV41Model.Config)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    defaults = _v41_optimizer_config(config.model)
    config.optim.optimizer._muon_profile = defaults._muon_profile
    config.optim.optimizer._optimizer_defaults["Muon"] = defaults._optimizer_defaults["Muon"]
    config.checkpointer = DeepSeekV41LoRACheckpointManager.Config()
    return config


def deepseek_v4_1_flash_cpt() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    # The registered converter promotes this upstream schema before CLI parsing.
    cast("Trainer.Config", config).checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash_40layers_16experts_multimodal()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_sft_compiled() -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    cast("Trainer.Config", config).checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft_compiled() -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_40layers_16experts_multimodal_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft_compiled() -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config
