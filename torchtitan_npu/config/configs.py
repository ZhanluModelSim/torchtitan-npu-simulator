# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed NPU extensions to TorchTitan's training configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal

import tyro
from torchtitan.components.optim import Optim, OptimizersContainer
from torchtitan.config import TrainingConfig as _BaseTrainingConfig
from torchtitan.config.parallelism import ParallelismConfig as _BaseParallelismConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout
from torchtitan.observability.profiler import Profiler as _BaseProfiler

QuantizationRecipe = Literal["all_mxfp8", "all_hif8", "mix", "all_block_fp8"]
LIQuantization = Literal["mxfp4", "mxfp8", "fp8", "hif8"]
KVNormQuantization = Literal["mxfp8"]


@dataclass(kw_only=True, slots=True)
class ModelCompileConfig:
    """NPU block compilation, applied by the model before FSDP wrapping."""

    backend: str = "inductor"
    fullgraph: bool = True


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    """Upstream non-model compilation plus an explicit NPU model policy."""

    components: list[str] = field(default_factory=list)
    model: Annotated[ModelCompileConfig | None, tyro.conf.AvoidSubcommands] = None


@dataclass(frozen=True, slots=True)
class MuonOptimizerProfile:
    """Model-owned metadata required to construct DistMuon.

    Scalar hyperparameters belong to the recipe's typed optimizer configs.
    """

    muon_pattern: str
    compute_sharding_by_fqn: Mapping[str, ComputeLayout]
    bucket_configs: tuple[BucketConfig, ...]


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(OptimizersContainer.Config):
    """Final typed optimizer groups, with the NPU CPU-offload carrier."""

    _cpu_offload: Annotated[bool, tyro.conf.Suppress] = False


@dataclass(kw_only=True, slots=True)
class OptimConfig(Optim.Config):
    """Optimization policy with the NPU optimizer container config."""

    optimizer: OptimizersContainer.Config = field(default_factory=OptimizerConfig)


@dataclass(kw_only=True, slots=True)
class KVNormQuantizationConfig:
    """KV norm fake quantization: the format and the norm sites to wrap.

    Always present on the quantization config (so the CLI stays
    ``--extension.quantization.kv-norm-quantization.<field>``); ``format=None``
    disables the transform.
    """

    format: KVNormQuantization | None = None
    """``mxfp8`` fake-quantizes the nope prefix (the channels RoPE leaves untouched)
    with MX block scales. ``None`` disables it.
    """
    fqns: list[str] = field(default_factory=list)
    """Config-tree FQN suffixes of the norm nodes to fake-quantize.

    A node is selected when its FQN is one of the suffixes or ends with
    ``"." + suffix``. Empty by default (nothing is selected): the
    launch scripts state the sites explicitly.
    """
    block_size: int = 32
    """MX block size of the fake quantization.
    """

    def __post_init__(self) -> None:
        if self.format not in (None, "mxfp8"):
            raise ValueError("format must be None or one of: mxfp8")


@dataclass(kw_only=True, slots=True)
class QuantizationExtensionConfig:
    """TorchAO-NPU quantized-training options.

    These fields define the public CLI schema. The quantization integration can
    consume them after CLI parsing without adding model-specific options to the
    upstream TorchTitan configuration.
    """

    enable_quantized_training: bool = False
    enable_sparse_attention_quantization: bool = False
    recipe: QuantizationRecipe = "all_block_fp8"
    enable_mxfp4_qat: bool = False
    li_quantization: LIQuantization | None = None
    """LI Q/K format.

    MXFP4/MXFP8 use MX block scales, FP8 uses per-token-head scales, and HiF8
    uses per-tensor scales.
    """
    dst_type_max: float = 0.0
    enable_fsdp_prequantize: bool = False
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"] = "fp8"
    """Routed-expert communication format when FSDP prequantization is enabled.

    MXFP4 requires MXFP4 QAT. Selected dense weights retain FP8 communication
    under the ``all_block_fp8`` recipe.
    """
    fsdp_prequantize_fqns: list[str] | None = None
    """FQN suffix patterns that ENABLE FSDP pre-quantize, only meaningful with
    ``enable_fsdp_prequantize=True``. ``None`` (unset) falls back to the recipe
    default whitelist; only whitelisted weights are enabled, non-whitelisted
    weights are explicitly disabled. A whitelisted weight whose shard breaks
    the pre-quantize alignment contract fails fast at runtime with a fix hint.
    An explicitly empty list with the master switch on is a contradictory
    configuration and raises ``ValueError``.
    """
    kv_norm_quantization: KVNormQuantizationConfig = field(default_factory=KVNormQuantizationConfig)
    enable_hif8_save_quant_codes: bool = False
    """Keep HiF8's quantized operands and result across the
    selective-AC boundary instead of re-deriving them in the backward. Off
    by default (bit-identical to not patching the save list at all)."""
    save_block_ops_level: int = 0
    """Add attention/mHC/MoE-routing NPU ops to the selective-AC MUST_SAVE
    list (1-3, cumulative; recipe-independent). Off (0) by default."""

    def validate(self) -> None:
        if self.fsdp_prequantize_dtype not in ("fp8", "mxfp4"):
            raise ValueError("fsdp_prequantize_dtype must be fp8 or mxfp4")
        if self.li_quantization not in (None, "mxfp4", "mxfp8", "fp8", "hif8"):
            raise ValueError("li_quantization must be None or one of: mxfp4, mxfp8, fp8, hif8")
        if self.enable_fsdp_prequantize and self.fsdp_prequantize_fqns is not None and not self.fsdp_prequantize_fqns:
            raise ValueError(
                "fsdp_prequantize_fqns is explicitly empty while enable_fsdp_prequantize is enabled; "
                "unset it to use the recipe default whitelist, or list the FQN patterns to enable."
            )


@dataclass(kw_only=True, slots=True)
class ExtensionConfig:
    """Global NPU extensions without an upstream component owner.

    Add a semantic group as a nested dataclass, then expose it with
    ``field(default_factory=...)``. For example::

        @dataclass(kw_only=True, slots=True)
        class RuntimeExtensionConfig:
            enable_feature: bool = False

        @dataclass(kw_only=True, slots=True)
        class ExtensionConfig:
            runtime: RuntimeExtensionConfig = field(
                default_factory=RuntimeExtensionConfig,
            )

    This produces the CLI option ``--extension.runtime.enable-feature``.
    """

    quantization: QuantizationExtensionConfig = field(
        default_factory=QuantizationExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class TrainingExtensionConfig:
    """NPU extensions owned by the training configuration."""

    allow_hf32: bool = True
    """Enable HF32 for the NPU matmul, convolution, and ACLNN backends."""


@dataclass(kw_only=True, slots=True)
class TrainingConfig(_BaseTrainingConfig):
    """Training options that are specific to NPU execution."""

    extension: TrainingExtensionConfig = field(
        default_factory=TrainingExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class ProfilerExtensionConfig:
    """NPU-specific options for the profiler component."""

    profiler_start: int | None = None
    """Absolute first training step to profile, inclusive."""

    profiler_end: int | None = None
    """Absolute training step at which profiling stops, exclusive."""

    profile_ranks: list[int] = field(default_factory=lambda: [-1])
    """Ranks to profile. ``[-1]`` profiles every rank."""

    profile_with_memory: bool = False
    """Whether to record memory events in the profiler trace."""

    profile_with_stack: bool = False
    """Whether to record Python/C++ stack information in the trace."""

    enable_online_parse: bool = True
    """Whether CANN should parse traces online via its trace handler."""


@dataclass(kw_only=True, slots=True)
class ProfilerConfig(_BaseProfiler.Config):
    """TorchTitan profiler configuration with NPU-specific extensions."""

    extension: ProfilerExtensionConfig = field(
        default_factory=ProfilerExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class ParallelismConfig(_BaseParallelismConfig):
    """TorchTitan parallelism configuration with NPU-specific extensions."""

    disable_expert_replication: bool = False
    """Shard routed experts across dense HSDP replicas so experts keep replica degree 1."""

    def __post_init__(self) -> None:
        base_post_init = getattr(_BaseParallelismConfig, "__post_init__", None)
        if base_post_init is not None:
            base_post_init(self)
        if not self.disable_expert_replication:
            return
        if self.data_parallel_replicate_degree <= 1:
            raise ValueError("disable_expert_replication requires data_parallel_replicate_degree > 1")
        if self.expert_parallel_degree <= 1:
            raise ValueError("disable_expert_replication requires expert_parallel_degree > 1")
        if self.tensor_parallel_degree != 1 or self.context_parallel_degree != 1 or self.pipeline_parallel_degree != 1:
            raise ValueError("disable_expert_replication currently requires TP=CP=PP=1")
