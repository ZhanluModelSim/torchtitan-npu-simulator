# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from torchtitan_npu.config import manager as config_manager
from torchtitan_npu.config.configs import ExtensionConfig, OptimizerConfig, TrainingConfig
from torchtitan_npu.config.converters import TrainerConfigConverter
from torchtitan_npu.distributed.utils import set_allow_hf32


class TrainerEx(Trainer):
    """Base trainer for NPU-specific training features."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        extension: ExtensionConfig = field(default_factory=ExtensionConfig)
        optimizer: OptimizerConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=OptimizerConfig,
        )
        training: TrainingConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=TrainingConfig,
        )

        def __post_init__(self) -> None:
            # ``slots=True`` dataclasses are recreated by the decorator, so a
            # zero-argument ``super()`` can retain the pre-decoration class cell.
            Trainer.Config.__post_init__(self)
            self.optimizer.materialize()
            if self.optimizer.name == "Muon" and (
                self.parallelism.tensor_parallel_degree > 1 or self.parallelism.pipeline_parallel_degree > 1
            ):
                raise ValueError(
                    "DeepSeek-V4 DistributedMuon requires "
                    "tensor_parallel_degree=1 and pipeline_parallel_degree=1; "
                    "TP _StridedShard and PP stage-local parameter groups are not admitted yet"
                )

    def __init__(self, config: Config):
        quantization_config = config.extension.quantization
        if quantization_config.enable_quantized_training:
            from interfaces.torchao_converter import apply_quantization_converter

            logger.info(
                "Applying TorchAO-NPU quantization recipe=%s before Trainer initialization",
                quantization_config.recipe,
            )
            model_compile_enabled = config.compile.enable and "model" in config.compile.components
            config.model_spec = apply_quantization_converter(
                config.model_spec,
                quantization_config,
                model_compile_enabled=model_compile_enabled,
            )

        set_allow_hf32(config.training.extension.allow_hf32)
        super().__init__(config)


config_manager.register_config_converter(
    Trainer.Config,
    TrainerConfigConverter(
        target_type=TrainerEx.Config,
        component_types={
            "optimizer": OptimizerConfig,
            "training": TrainingConfig,
        },
    ),
)
