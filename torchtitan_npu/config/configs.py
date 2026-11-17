# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed NPU extensions to TorchTitan's training configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import tyro
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import TrainingConfig as _BaseTrainingConfig
from torchtitan.tools.profiler import Profiler as _BaseProfiler

QuantizationRecipe = Literal["all_mxfp8", "all_hif8", "mix", "all_block_fp8"]
LIQuantization = Literal["mxfp4", "mxfp8", "fp8", "hif8"]
KVNormQuantization = Literal["mxfp8"]


@dataclass(frozen=True, slots=True)
class MuonOptimizerProfile:
    """Model-owned metadata required to construct DistMuon.

    The profile intentionally excludes scalar optimizer hyperparameters. Those
    are public CLI fields on :class:`OptimizerConfig` and are materialized only
    after Tyro has applied command-line overrides.
    """

    muon_pattern: str
    optimizer_factory_kwargs: Mapping[str, Mapping[str, Any]]


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(OptimizersContainer.Config):
    """NPU optimizer CLI schema while preserving recipe-provided AdamW configs."""

    name: Literal["AdamW", "Muon"] = "AdamW"
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_enable_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw", "spectral_unclamped"] = "match_rms_adamw"
    muon_ns_coefficients: tuple[float, float, float] = (
        3.4445,
        -4.7750,
        2.0315,
    )
    muon_eps: float = 1e-7
    muon_enable_hsdp_replica_dedup: bool = False
    """Assign each DistMuon tensor state to one HSDP replica domain."""
    _muon_profile: Annotated[MuonOptimizerProfile | None, tyro.conf.Suppress] = None
    _cpu_offload: Annotated[bool, tyro.conf.Suppress] = False
    """Carrier for ``--training.enable-cpu-offload``, set by the trainer
    config so the optimizer-container override can branch on it; not a
    user-facing switch."""

    def materialize(self) -> None:
        """Turn an explicit Muon selection into upstream optimizer groups.

        ``AdamW`` is intentionally a strict no-op so converting every NPU
        recipe to this schema cannot alter its existing optimizer behavior.
        """
        if self.name == "AdamW":
            return
        if self._muon_profile is None:
            raise ValueError("optimizer.name=Muon requires a recipe with a DSV4 Muon profile")

        self.param_groups = [
            ParamGroupConfig(
                pattern=self._muon_profile.muon_pattern,
                optimizer_name="DistMuon",
                optimizer_kwargs={
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "momentum": self.muon_momentum,
                    "nesterov": self.muon_enable_nesterov,
                    "ns_steps": self.muon_ns_steps,
                    "adjust_lr_fn": self.muon_adjust_lr_fn,
                    "ns_coefficients": self.muon_ns_coefficients,
                    "eps": self.muon_eps,
                    "fused": False,
                    "foreach": False,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": self.lr,
                    "betas": (self.beta1, self.beta2),
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                    "fused": True,
                    "foreach": False,
                },
            ),
        ]
        self.optimizer_factory_kwargs_by_name = {
            name: dict(kwargs) for name, kwargs in self._muon_profile.optimizer_factory_kwargs.items()
        }
        self.optimizer_factory_kwargs_by_name["DistMuon"]["enable_hsdp_replica_dedup"] = (
            self.muon_enable_hsdp_replica_dedup
        )


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
