# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for MAGI-2-preview."""

from dataclasses import dataclass, field

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    CheckpointConfig,
    OptimizerConfig,
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig as NpuTrainerConfig,
    TrainingConfig,
)

from . import model_registry
from .config_overrides import (
    Magi2PreviewModelOverrides,
    apply_model_overrides,
    build_model_spec_with_overrides,
)
from .dataset import Magi2SyntheticDataLoader
from .latent_dataset import Magi2LatentDataLoader


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """MAGI-2-preview config with stable model CLI overrides."""

    model_overrides: Magi2PreviewModelOverrides = field(
        default_factory=Magi2PreviewModelOverrides
    )

    def __post_init__(self) -> None:
        self.model_spec = apply_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _parallelism() -> ParallelismConfig:
    """Return the MAGI-2-preview baseline parallel layout.

    The baseline exercises FSDP sharding only; the other degrees are kept
    at 1 but are implemented and can be enabled via CLI:

    - ``tensor_parallel_degree``: sequence-replicated TP v1 (head/column
      splits with all-reduced outputs; TP+CP and TP+EP raise for now).
    - ``pipeline_parallel_degree``: ``pipeline_magi2`` stage splitting
      (v1: single microbatch, GPipe for pp>1; PP+CP/TP/EP raise).
    - ``context_parallel_degree``: Ulysses CP (sequence shards in original
      token order; CP+EP raises until the combined head mesh lands).

    ``expert_parallel_degree`` is wired but kept at 1 in the baseline:
    setting it above 1 enables head-parallel MoE
    (``parallelize._apply_moe_parallel``), which shards every routed-MoE
    layer along the head axis and requires ``moe_num_heads %
    expert_parallel_degree == 0``. That regime (a) assumes replicated
    tokens (zero-padded partial outputs all-reduced over the EP mesh); the
    Ulysses seq<->head all-to-all regime needs context parallelism, so EP
    beyond it awaits the CP+EP combination.
    ``expert_tensor_parallel_degree`` must stay 1 (raises with TP/EP).
    """
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
        context_parallel_degree=1,
    )


def _trainer_config(
    *,
    flavor: str,
    training: TrainingConfig,
    optimizer: OptimizerConfig,
    lr_scheduler: LRSchedulersContainer.Config,
    parallelism: ParallelismConfig,
    activation_checkpoint: ActivationCheckpointConfig,
    dataloader: Magi2SyntheticDataLoader.Config | Magi2LatentDataLoader.Config,
    print_config: bool,
) -> TrainerConfig:
    model_spec, model_overrides = build_model_spec_with_overrides(
        model_registry(flavor)
    )
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
        debug=DebugConfig(print_config=print_config),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(converters=[]),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=training,
        parallelism=parallelism,
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=activation_checkpoint,
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def magi2_preview_smoketest() -> TrainerConfig:
    """Minimal debug-flavor recipe on synthetic latents for local bring-up."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            # Synthetic samples pack to a fixed token count
            # (2*4*4 video + 16 audio + 16 text); seq_len only feeds MFU.
            seq_len=64,
            max_norm=1.0,
            steps=2,
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        dataloader=Magi2SyntheticDataLoader.Config(
            video_frames=2,
            video_height=4,
            video_width=4,
            audio_len=16,
            text_len=16,
        ),
        print_config=True,
    )


def magi2_preview_latent_smoketest() -> TrainerConfig:
    """Debug-flavor recipe on pre-encoded latents for local bring-up.

    Identical to :func:`magi2_preview_smoketest` but streams from an offline
    latent shard directory via :class:`Magi2LatentDataLoader` instead of
    synthetic samples. ``data_path`` defaults to the directory written by
    ``scripts/magi2_preprocess_latents.py --dry-run``; point it at your own
    pre-encoded shards (see docs/user-guides/magi2_preview_data_pipeline.md)
    for real training. Kept out of the simulator registry on purpose: it
    needs a materialized shard directory to build.
    """
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            # Packs hold up to max_tokens_per_pack latent tokens; seq_len
            # only feeds MFU accounting.
            seq_len=64,
            max_norm=1.0,
            steps=2,
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=1e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        dataloader=Magi2LatentDataLoader.Config(
            data_path="./magi2_latent_shards",
            max_tokens_per_pack=4096,
            seed=0,
        ),
        print_config=True,
    )


def magi2_preview_baseline_bf16() -> TrainerConfig:
    """Full MAGI-2-preview baseline with FSDP auto-sharding.

    The synthetic shapes and step schedule are scaffolding until a real
    latent dataset (VAE-encoded video/audio + text embeddings) exists; treat
    them as placeholders, not a converged recipe.
    """
    return _trainer_config(
        flavor="full",
        training=TrainingConfig(
            local_batch_size=1,
            # 8*16*16 video + 64 audio + 128 text packed tokens; seq_len
            # only feeds MFU since samples are packed to a fixed size.
            seq_len=2240,
            max_norm=1.0,
            steps=2000,
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=_parallelism(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        dataloader=Magi2SyntheticDataLoader.Config(
            video_frames=8,
            video_height=16,
            video_width=16,
            audio_len=64,
            text_len=128,
        ),
        print_config=True,
    )
