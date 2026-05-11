# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch.nn as nn

from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter
from torchtitan.trainer import Trainer

from torchtitan_npu.converters.model_custom_interface import ModelCustomConfig
from torchtitan_npu.converters.npu_registry import get_using_train_spec

from .parallelize_plan_update_wrapper import apply_parallelize_plan_update
from .state_dict_update_wrapper import apply_state_dict_update

logger = logging.getLogger(__name__)


class ModelCustomConfigConverter(ModelConverter):

    _model_config: ModelCustomConfig

    def __init__(self, trainer_config: Trainer.Config, parallel_dims: ParallelDims):
        self.trainer_config = trainer_config
        self.parallel_dims = parallel_dims
        self.model_name = trainer_config.model_spec.name

    def convert(self, model: nn.Module):
        try:
            logger.info(
                f"[ModelCustomConfigConverter] Applied '{self._model_config.name}' start ..."
            )

            train_spec = get_using_train_spec()
            if train_spec is None:
                raise RuntimeError(
                    "[ModelCustomConfigConverter] CUR_USING_TRAIN_SPEC is not set."
                )

            model_converter = self._model_config.model_converter
            if model_converter is not None:
                model_converter(self.trainer_config, self.parallel_dims).convert(model)
                logger.info(f"[ModelCustomConfigConverter] Applied {model_converter}.")

            parallelize_plan_updater = self._model_config.parallelize_plan_updater
            if parallelize_plan_updater is not None:
                apply_parallelize_plan_update(parallelize_plan_updater)
                logger.info(
                    f"[ModelCustomConfigConverter] Applied {parallelize_plan_updater}."
                )

            state_dict_updater = self._model_config.state_dict_updater
            if state_dict_updater is not None:
                apply_state_dict_update(state_dict_updater, train_spec)
                logger.info(
                    f"[ModelCustomConfigConverter] Applied {state_dict_updater}."
                )

            logger.info(
                f"[ModelCustomConfigConverter] Applied '{self._model_config.name}' end."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply custom model config '{self._model_config.name}' : {e}"
            ) from e
