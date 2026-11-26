# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 model classes and flavor configuration entry points."""

from .flavors import MODEL_FLAVORS, EngramArgs, build_model_config
from .model import DeepSeekV41Model, DeepSeekV41MultimodalModel, DeepSeekV41TransformerBlock
from .state_dict_adapter import DeepSeekV41StateDictAdapter

DeepSeekV41Model.state_dict_adapter_cls = DeepSeekV41StateDictAdapter

__all__ = [
    "MODEL_FLAVORS",
    "DeepSeekV41Model",
    "DeepSeekV41MultimodalModel",
    "DeepSeekV41TransformerBlock",
    "EngramArgs",
    "build_model_config",
]
