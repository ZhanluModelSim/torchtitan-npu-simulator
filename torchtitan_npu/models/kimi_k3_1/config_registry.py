# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native training factories for the independent Kimi K3.1 family."""

from torchtitan.trainer import Trainer

from torchtitan_npu.models.kimi_k3.config_registry import _trainer_config as _k3_trainer_config

from . import kimi_k3_1_configs
from .state_dict_adapter import KimiK31StateDictAdapter


def _trainer_config(flavor: str, *, num_mtp_layers: int = 0) -> Trainer.Config:
    trainer = _k3_trainer_config(flavor, num_mtp_layers=num_mtp_layers)
    spec = trainer.model_spec
    assert spec is not None
    spec.name = "kimi_k3_1"
    spec.model = kimi_k3_1_configs[flavor]()
    spec.state_dict_adapter = KimiK31StateDictAdapter
    return trainer


def kimi_k3_1_debugmodel(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _trainer_config("debug", num_mtp_layers=num_mtp_layers)


def kimi_k3_1_full(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _trainer_config("full", num_mtp_layers=num_mtp_layers)


def kimi_k3_1_16layer_reduced(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _trainer_config("16layer_reduced", num_mtp_layers=num_mtp_layers)
