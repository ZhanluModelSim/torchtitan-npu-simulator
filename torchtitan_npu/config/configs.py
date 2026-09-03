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

QuantizationRecipe = Literal["all_mxfp8", "mix", "all_block_fp8"]


@dataclass(frozen=True, slots=True)
class MuonOptimizerProfile:
    """Model-owned metadata required to construct DistributedMuon.

    The profile intentionally excludes scalar optimizer hyperparameters. Those
    are public CLI fields on :class:`OptimizerConfig` and are materialized only
    after Tyro has applied command-line overrides.
    """

    muon_pattern: str
    optimizer_factory_kwargs: Mapping[str, Mapping[str, Any]]


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(OptimizersContainer.Config):
    """NPU optimizer CLI schema while preserving native optimizer configs."""

    name: Literal["native", "Muon"] = "native"
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
    _muon_profile: Annotated[MuonOptimizerProfile | None, tyro.conf.Suppress] = None

    def materialize(self) -> None:
        """Turn an explicit Muon selection into upstream optimizer groups.

        ``native`` is intentionally a strict no-op so converting every NPU
        recipe to this schema cannot alter its existing optimizer behavior.
        """
        if self.name == "native":
            return
        if self._muon_profile is None:
            raise ValueError("optimizer.name=Muon requires a recipe with a DSV4 Muon profile")

        self.param_groups = [
            ParamGroupConfig(
                pattern=self._muon_profile.muon_pattern,
                optimizer_name="DistributedMuon",
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
                    "fused": False,
                    "foreach": False,
                },
            ),
        ]
        self.optimizer_factory_kwargs_by_name = {
            name: dict(kwargs) for name, kwargs in self._muon_profile.optimizer_factory_kwargs.items()
        }


@dataclass(kw_only=True, slots=True)
class QuantizationExtensionConfig:
    """TorchAO-NPU quantized-training options.

    These fields define the public CLI schema. The quantization integration can
    consume them after CLI parsing without adding model-specific options to the
    upstream TorchTitan configuration.
    """

    enable_quantized_training: bool = False
    recipe: QuantizationRecipe = "mix"
    enable_mxfp4_qat: bool = False
    dst_type_max: float = 0.0


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
