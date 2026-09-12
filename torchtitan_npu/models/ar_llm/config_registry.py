# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for ar_llm (DeepSeekV4-Sparse)."""

from dataclasses import dataclass, field

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.quantization.mx import MXFP8Converter
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
    ArLlmModelOverrides,
    apply_model_overrides,
    build_model_spec_with_overrides,
)


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """ar_llm config with stable model CLI overrides."""

    model_overrides: ArLlmModelOverrides = field(default_factory=ArLlmModelOverrides)
    mxfp8_fqns: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        self.model_spec = apply_model_overrides(
            self.model_spec,
            self.model_overrides,
        )
        _apply_mxfp8_fqns_override(self.model_converters, self.mxfp8_fqns)


def _apply_mxfp8_fqns_override(
    model_converters: ModelConvertersContainer.Config,
    fqns: list[str] | None,
) -> None:
    if fqns is None:
        return

    mxfp8_configs = [
        converter
        for converter in model_converters.converters
        if isinstance(converter, MXFP8Converter.Config)
    ]
    if len(mxfp8_configs) != 1:
        raise ValueError(
            "mxfp8_fqns requires exactly one MXFP8 converter, "
            f"but found {len(mxfp8_configs)}"
        )
    mxfp8_configs[0].fqns = list(fqns)


# MXFP8 quantization scope by module-fqn substring. The attention entries
# cover every executed attention matmul: the MLA q/kv projections
# (``attention.q_proj``/``attention.kv_proj``, direct ArLlmAttention children),
# the HCA indexer (``attention.core.indexer_*``) and all KDA projections
# (``attention.kda.*``). The grouped O projection (``attention.o_proj``) is
# excluded: it is an einsum on 3D parameters whose contraction does not match
# the linear-like pattern handled by the wrapper einsum patch
# (patches/torchao_npu/mxfp8_wrapper_einsum.py), so it would fall back to BF16.
# ``moe.shared_experts`` einsums ARE covered by that patch.
DEFAULT_MXFP8_FQNS = [
    "moe.experts",
    "moe.shared_experts",
    "attention.q_proj",
    "attention.kv_proj",
    "attention.core",
    "attention.kda",
]


def _default_converters(*, enable_mxfp8: bool) -> list:
    """npu_rms_norm/npu_rope have simulator shims; model-specific fused ops
    (KDA, CSA/HCA, mHC, LatentMoE GMM) are captured through their own paths.
    MXFP8 quantization is opt-in per flavor via ``enable_mxfp8``; the covered
    fqns can be overridden with ``--mxfp8-fqns``."""
    converters = [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]
    if enable_mxfp8:
        converters.append(
            MXFP8Converter.Config(
                recipe_name="mxfp8_rceil",
                fqns=list(DEFAULT_MXFP8_FQNS),
            )
        )
    return converters


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
    enable_mxfp8: bool = False,
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
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
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


def ar_llm_debug() -> TrainerConfig:
    """Minimal 6-layer (one CSA/HCA/KDA unit) recipe for local debugging."""
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


def ar_llm_debug_mxfp8() -> TrainerConfig:
    """ar_llm_debug with MXFP8 quantized MoE + attention projections."""
    return _trainer_config(
        flavor="debug",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        enable_mxfp8=True,
    )


def ar_llm_reduced() -> TrainerConfig:
    """12-layer reduced spec for meta single-step and AC toggle tests."""
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


def ar_llm_reduced_mxfp8() -> TrainerConfig:
    """ar_llm_reduced with MXFP8 quantized MoE + attention projections."""
    return _trainer_config(
        flavor="reduced",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        enable_mxfp8=True,
    )


def ar_llm_50t() -> TrainerConfig:
    """Official 50T spec (meta/capacity validation only, never real training)."""
    return _trainer_config(
        flavor="50t",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=65536,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def ar_llm_100t() -> TrainerConfig:
    """Official 100T spec (meta/capacity validation only, never real training)."""
    return _trainer_config(
        flavor="100t",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=65536,
            max_norm=1.0,
            steps=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )
