# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flavors import (
    MODEL_FLAVORS,
    Attention,
    DeepSeekV4Model,
    DeepSeekV4TransformerBlock,
    HcPost,
    HcPre,
    MTPLoss,
    build_model_config,
)
from .state_dict_adapter import DeepSeekV4StateDictAdapter

DeepSeekV4Model.state_dict_adapter_cls = DeepSeekV4StateDictAdapter

__all__ = [
    "MODEL_FLAVORS",
    "Attention",
    "DeepSeekV4Model",
    "DeepSeekV4TransformerBlock",
    "HcPost",
    "HcPre",
    "MTPLoss",
    "build_model_config",
]
