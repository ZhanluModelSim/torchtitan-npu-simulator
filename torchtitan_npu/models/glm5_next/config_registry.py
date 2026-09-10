# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for glm5_next (GLM-5.3-Flash)."""

from dataclasses import dataclass, field

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    CheckpointConfig,
    OptimizerConfig,
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig as NpuTrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry
from .config_overrides import (
    Glm5NextModelOverrides,
    apply_model_overrides,
    build_model_spec_with_overrides,
)
from .mm_data import Glm5NextMultiModalDataLoader


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """glm5_next config with stable model CLI overrides."""

    model_overrides: Glm5NextModelOverrides = field(default_factory=Glm5NextModelOverrides)

    def __post_init__(self) -> None:
        self.model_spec = apply_model_overrides(
            self.model_spec,
            self.model_overrides,
        )


def _default_converters() -> list:
    """npu_rms_norm covers norms; npu_mhc_pre/post fuse the mHC sites (the
    simulator patches them into shape-only shims via apply_mhc_shims). The
    remaining model-specific fused ops (KDA conv/chunk_kda, DSA
    indexer/sparse attention) are bound by ``apply_glm5_next_shims`` in the
    simulator (MODEL_CONTRACT.md section 11)."""
    return [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_mhc_pre"),
        get_model_converter_config("npu_mhc_post"),
    ]


def _parallelism(*, expert_parallel_degree: int = 1) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=expert_parallel_degree,
        expert_tensor_parallel_degree=1,
        context_parallel_degree=1,
    )


def _trainer_config(
    *,
    flavor: str,
    training: TrainingConfig,
    activation_checkpoint: ActivationCheckpointConfig,
) -> TrainerConfig:
    model_spec, model_overrides = build_model_spec_with_overrides(
        model_registry(flavor)
    )
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/deepseekv3_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
        debug=DebugConfig(print_config=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters()
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
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
        training=training,
        parallelism=_parallelism(),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=activation_checkpoint,
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def glm5_next_debug() -> TrainerConfig:
    """Minimal 9-block (4 pre + 1 looped x2 + 4 post) recipe for local debugging."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def glm5_next_reduced() -> TrainerConfig:
    """12-block reduced spec for meta single-step and AC toggle tests."""
    return _trainer_config(
        flavor="reduced",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def glm5_next_full() -> TrainerConfig:
    """Official 96-layer spec (meta/capacity validation only, never real training)."""
    return _trainer_config(
        flavor="full",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def glm5_next_debug_mm() -> TrainerConfig:
    """Debug spec with the cc12m-test multimodal channel (uniform square grid).

    Uses the vlm tokenizer whose ``<|image|>`` special token id (1998) must
    match the model's ``image_token_id``. Vision v1 requires a uniform square
    grid, so the loader forces ``image_size=56`` (4x4 patches, 4 merged
    tokens per image); see MODEL_CONTRACT.md section 7.
    """
    model_spec, model_overrides = build_model_spec_with_overrides(model_registry("debug"))
    model_overrides.image_token_id = 1998
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/vlm_tokenizer",
        model_spec=model_spec,
        model_overrides=model_overrides,
        debug=DebugConfig(print_config=True),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters()
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=Glm5NextMultiModalDataLoader.Config(
            dataset="cc12m-test",
            patch_size=14,
            spatial_merge_size=2,
            max_patches_per_image=16,
            max_images_per_batch=2,
            image_size=56,
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
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        parallelism=_parallelism(),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )
