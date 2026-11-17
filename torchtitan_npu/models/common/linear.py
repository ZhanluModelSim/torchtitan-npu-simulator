# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from dataclasses import dataclass

import torch.nn.functional as F
from torchtitan.models.common.linear import Linear


class SharedLinear(Linear):
    """A shared projection that can preserve input gradients without weight gradients."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    _detach_weight = False

    @contextmanager
    def detached_weight(self):
        previous = self._detach_weight
        self._detach_weight = True
        try:
            yield
        finally:
            self._detach_weight = previous

    def forward(self, input):
        if not self._detach_weight:
            return super().forward(input)
        bias = self.bias.detach() if self.bias is not None else None
        return F.linear(input, self.weight.detach(), bias)
