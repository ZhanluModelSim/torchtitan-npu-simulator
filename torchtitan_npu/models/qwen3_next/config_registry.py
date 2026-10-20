# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for Qwen3-Next."""

from dataclasses import dataclass, field

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.config import ActivationCheckpointConfig, CommConfig, CompileConfig, DebugConfig
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


# MXFP8 FQNs are substring-matched against module FQNs by the upstream
# converter (``target_fqn in cur_fqn``), so they must be precise enough to
# never capture a router gate. Quantized in v1: the dense token-mixer
# projections (q/k/v/o, GDN in_proj/out_proj) and the shared-expert MLP.
# The routed grouped experts keep raw parameters (torch._grouped_mm path,
# outside the torchao nn.Linear wrapping) and stay bf16 - declared in the
# model contract. Router gates stay unquantized (fp32 semantics).
_MXFP8_FQNS = [
    "token_mixer.q_proj",
    "token_mixer.k_proj",
    "token_mixer.v_proj",
    "token_mixer.o_proj",
    "token_mixer.in_proj_qkvz",
    "token_mixer.in_proj_ba",
    "token_mixer.out_proj",
    "moe.shared_expert.gate_proj",
    "moe.shared_expert.up_proj",
    "moe.shared_expert.down_proj",
]

# Router-style parameters are force-excluded from quantization (fp32).
_ROUTER_FQNS = ("moe.gate", "moe.shared_expert_gate")


def _guard_mxfp8_fqns(fqns: list[str]) -> None:
    """Fail fast when a configured FQN would capture a router gate."""
    for fqn in fqns:
        for router in _ROUTER_FQNS:
            if fqn in router:
                raise ValueError(
                    f"MXFP8 fqn '{fqn}' would capture the router gate "
                    f"'{router}'; router parameters are force-excluded "
                    "from quantization"
                )


@dataclass(kw_only=True, slots=True)
class TrainerConfig(NpuTrainerConfig):
    """Qwen3-Next config with MXFP8 FQN CLI overrides."""

    mxfp8_fqns: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
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
    _guard_mxfp8_fqns(list(fqns))
    mxfp8_configs[0].fqns = list(fqns)


def _default_converters(*, enable_mxfp8: bool = False) -> list:
    converters = [
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
    ]
    if enable_mxfp8:
        _guard_mxfp8_fqns(_MXFP8_FQNS)
        converters.append(
            MXFP8Converter.Config(
                recipe_name="mxfp8_rceil",
                fqns=list(_MXFP8_FQNS),
            )
        )
    return converters


def _trainer_config(
    *,
    flavor: str,
    training: TrainingConfig,
    optimizer: OptimizerConfig,
    lr_scheduler: LRSchedulersContainer.Config,
    parallelism: ParallelismConfig,
    activation_checkpoint: ActivationCheckpointConfig,
    print_config: bool,
    enable_mxfp8: bool = False,
) -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path="./tests/assets/tokenizer/qwen3-tokenizer",
        model_spec=model_registry(flavor),
        debug=DebugConfig(print_config=print_config),
        comm=CommConfig(trace_buf_size=0),
        model_converters=ModelConvertersContainer.Config(
            converters=_default_converters(enable_mxfp8=enable_mxfp8)
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training=training,
        parallelism=parallelism,
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=activation_checkpoint,
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def _parallelism(
    *,
    expert_parallel_degree: int = 1,
    tensor_parallel_degree: int = 1,
) -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=tensor_parallel_degree,
        pipeline_parallel_degree=1,
        expert_parallel_degree=expert_parallel_degree,
        expert_tensor_parallel_degree=1,
        context_parallel_degree=1,
    )


def _2layer_config(
    *,
    tensor_parallel_degree: int = 1,
    expert_parallel_degree: int = 2,
    enable_mxfp8: bool = False,
) -> TrainerConfig:
    return _trainer_config(
        flavor="80B-A3B-2L",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=16384,
            max_norm=1.0,
            steps=1,
            # MindSpeed fsdp2 example uses reduce_dtype bf16
            mixed_precision_reduce="bfloat16",
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=_parallelism(
            expert_parallel_degree=expert_parallel_degree,
            tensor_parallel_degree=tensor_parallel_degree,
        ),
        # Measured config runs with recompute: False
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        print_config=True,
        enable_mxfp8=enable_mxfp8,
    )


def qwen3_next_smoketest() -> NpuTrainerConfig:
    """Minimal 4-layer recipe (3 GDN + 1 gated attention); local debugging."""
    return _trainer_config(
        flavor="debugmodel",
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            max_norm=1.0,
            steps=2,
            # MindSpeed fsdp2 example uses reduce_dtype bf16
            mixed_precision_reduce="bfloat16",
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
        # Measured-aligned tests run with ep=2 (admin round-6 directive)
        parallelism=_parallelism(expert_parallel_degree=2),
        # Measured fsdp2 config runs with recompute: False
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        print_config=True,
    )


def qwen3_next_80b_a3b_2layer() -> NpuTrainerConfig:
    """Measured comparison config: full 80B dims with only the layer
    count reduced to 2 (one full-attention + one linear-attention,
    both MoE). Defaults to EP=2 at the measured alignment."""
    return qwen3_next_80b_a3b(flavor="80B-A3B-2L")


def qwen3_next_80b_a3b(flavor: str = "80B-A3B") -> NpuTrainerConfig:
    """Qwen3-Next-80B-A3B baseline mirroring the measured fsdp2 run
    (pretrain_qwen3_next_16k_fsdp2_A3.yaml): FSDP=8, EP=2, eFSDP=4,
    recompute disabled, reduce bf16.

    Override via CLI, e.g. ``--parallelism.expert_parallel_degree 8``.
    """
    return _2layer_config(
        expert_parallel_degree=2,
        enable_mxfp8=False,
    ) if flavor == "80B-A3B-2L" else _full_config(flavor=flavor)


def _full_config(flavor: str = "80B-A3B") -> NpuTrainerConfig:
    return _trainer_config(
        flavor=flavor,
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=16384,
            max_norm=1.0,
            steps=1,
            mixed_precision_reduce="bfloat16",
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=_parallelism(expert_parallel_degree=2),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        print_config=True,
    )


# Acceptance-matrix variants (guide section 7; run with NGPU set to the
# world size of each cell). All use the 2-layer reduced flavor with the
# measured 80B dims, seq 16384, AC off; AC-on runs are produced by
# overriding --activation-checkpoint.mode full.


def qwen3_next_2l_tp2() -> NpuTrainerConfig:
    """TP=2 single-dimension cell (NGPU=2)."""
    return _2layer_config(tensor_parallel_degree=2, expert_parallel_degree=1)


def qwen3_next_2l_ep2() -> NpuTrainerConfig:
    """EP=2 single-dimension cell (NGPU=2)."""
    return _2layer_config(expert_parallel_degree=2)


def qwen3_next_2l_fsdp2() -> NpuTrainerConfig:
    """FSDP=2 single-dimension cell (NGPU=2; dp_shard=-1 auto-fills)."""
    return _2layer_config(expert_parallel_degree=1)


def qwen3_next_2l_tp2ep2() -> NpuTrainerConfig:
    """TP=2 x EP=2 pair combination (NGPU=4)."""
    return _2layer_config(tensor_parallel_degree=2, expert_parallel_degree=2)


def qwen3_next_2l_fsdp2ep2() -> NpuTrainerConfig:
    """FSDP=2 x EP=2 pair combination (NGPU=4)."""
    return _2layer_config(expert_parallel_degree=2)


def qwen3_next_2l_tp2ep2fsdp2() -> NpuTrainerConfig:
    """Core combination TP=2 x EP=2 x FSDP=2 (NGPU=8)."""
    return _2layer_config(tensor_parallel_degree=2, expert_parallel_degree=2)


def qwen3_next_2l_mxfp8() -> NpuTrainerConfig:
    """MXFP8 quantized variant (NGPU=1): quantized dense projections and
    shared experts enter the op stream (npu_dynamic_mx_quant /
    npu_quant_matmul); routers stay unquantized."""
    return _2layer_config(expert_parallel_degree=1, enable_mxfp8=True)
