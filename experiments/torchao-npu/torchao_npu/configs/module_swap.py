# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO module-level QAT configuration primitives."""

__all__ = ["ModuleSwapConfig"]

from dataclasses import dataclass
from typing import ClassVar

from torchao.quantization.qat import QATConfig, QATStep


@dataclass(kw_only=True, slots=True)
class ModuleSwapConfig(QATConfig):
    """Base config for transforms which replace a module's forward behavior."""

    base_config: ClassVar[None] = None
    activation_config: ClassVar[None] = None
    weight_config: ClassVar[None] = None
    step: QATStep = QATStep.PREPARE

    def __post_init__(self) -> None:
        self.step = QATStep(self.step)
