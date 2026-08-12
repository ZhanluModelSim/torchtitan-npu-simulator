# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass

import torch.nn as nn
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter

from torchtitan_npu.converters.model_custom_interface import ModelCustomConfig
from torchtitan_npu.patches.torchtitan._trainer_config_stash import (
    get_trainer_config,
)

from .parallelize_plan_update_wrapper import apply_parallelize_plan_update
from .state_dict_update_wrapper import apply_state_dict_update

logger = logging.getLogger(__name__)


class ModelCustomConfigConverter(Configurable, ModelConverter):
    _model_config: ModelCustomConfig

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.parallel_dims = parallel_dims
        self.model_compile_enabled = model_compile_enabled
        # pyrefly: ignore [missing-attribute]
        self.model_spec = get_trainer_config().model_spec
        self.model_name = self.model_spec.name

    def convert(self, model: nn.Module):
        try:
            logger.info(f"[ModelCustomConfigConverter] Applied {self._model_config.name!r} start ...")

            model_converter = self._model_config.model_converter
            if model_converter is not None:
                model_converter(self.model_spec).convert(model)
                logger.info(f"[ModelCustomConfigConverter] Applied {model_converter}.")

            parallelize_plan_updater = self._model_config.parallelize_plan_updater
            if parallelize_plan_updater is not None:
                apply_parallelize_plan_update(parallelize_plan_updater)
                logger.info(f"[ModelCustomConfigConverter] Applied {parallelize_plan_updater}.")

            state_dict_updater = self._model_config.state_dict_updater
            state_dict_updater_predicate = getattr(
                self._model_config,
                "state_dict_updater_predicate",
                None,
            )
            if state_dict_updater is not None and (
                state_dict_updater_predicate is None or state_dict_updater_predicate(model)
            ):
                apply_state_dict_update(state_dict_updater, self.model_spec)
                logger.info(f"[ModelCustomConfigConverter] Applied {state_dict_updater}.")

            logger.info(f"[ModelCustomConfigConverter] Applied {self._model_config.name!r} end.")
        except Exception as e:
            raise RuntimeError(f"Failed to apply custom model config {self._model_config.name!r} : {e}") from e
