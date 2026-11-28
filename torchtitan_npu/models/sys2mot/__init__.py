# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sys2MoT VLA (DeepSeek layer-paired Mixture-of-Transformers reference port)."""

from .config_registry import model_registry, sys2mot_debugmodel_multimodal
from .loss import Sys2MotJointLoss
from .model import Sys2MotModel

__all__ = ["Sys2MotJointLoss", "Sys2MotModel", "model_registry", "sys2mot_debugmodel_multimodal"]
