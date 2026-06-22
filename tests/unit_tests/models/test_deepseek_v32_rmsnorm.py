# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("torchtitan", reason="upstream torchtitan is required")


class _BuildConfig:
    def __init__(self, value):
        self.value = value

    def build(self):
        return self.value


def test_model_root_norm_honors_config_eps_and_param_init():
    from torchtitan.models.common import RMSNorm

    from torchtitan_npu.models.deepseek_v32.model import DeepSeekV32ModelNpu

    def init_weight(param: nn.Parameter) -> None:
        nn.init.constant_(param, 0.25)

    norm_config = RMSNorm.Config(
        normalized_shape=4,
        eps=0.123,
        param_init={"weight": init_weight},
    )
    model_config = SimpleNamespace(
        tok_embeddings=_BuildConfig(nn.Identity()),
        rope=_BuildConfig(SimpleNamespace(cache=torch.empty(0))),
        norm=norm_config,
        output=_BuildConfig(nn.Identity()),
        layers=[],
        num_mtp_modules=0,
    )

    model = DeepSeekV32ModelNpu(model_config)

    # The root norm is built straight from config.norm (upstream RMSNorm), so
    # its eps and param_init take effect.
    assert isinstance(model.norm, RMSNorm)
    assert model.norm.eps == norm_config.eps

    with torch.no_grad():
        model.norm.weight.zero_()
    model.norm.init_states()

    assert torch.allclose(model.norm.weight, torch.full_like(model.norm.weight, 0.25))
