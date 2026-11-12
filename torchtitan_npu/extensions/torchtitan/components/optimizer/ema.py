# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU EMA is deferred until the upstream multi-replica state is adapted."""

from dataclasses import dataclass

from torchtitan.components.optim.ema import EMA as _EMA


class EMA(_EMA):
    @dataclass(kw_only=True, slots=True)
    class Config(_EMA.Config):
        pass

    def __init__(self, config, **kwargs):
        raise NotImplementedError("NPU EMA is not supported in the eager upgrade")
