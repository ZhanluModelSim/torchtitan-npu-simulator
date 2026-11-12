# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TileLang router override selection through the real model config tree."""

import pytest
from torchtitan.config import OverrideConfig, apply_overrides, derive
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.moe import DeepSeekV41Router
from torchtitan_npu.override.deepseek_v4.moe_router import TilelangRouter, TilelangV4Router, TilelangV41Router


@pytest.mark.parametrize(
    ("router_type", "factory", "target_type"),
    [
        (TokenChoiceTopKRouter.Config, "tilelang", TilelangRouter.Config),
        (DeepSeekV4Router.Config, "tilelang_v4", TilelangV4Router.Config),
        (DeepSeekV41Router.Config, "tilelang_v41", TilelangV41Router.Config),
    ],
    ids=["generic", "v4", "v41"],
)
def test_tilelang_override_selects_only_its_router_config(router_type, factory, target_type):
    model = model_registry("deepseek_v4_1_debugmodel_text")
    layer = model.layers[0]
    original = layer.feed_forward.router
    if router_type is DeepSeekV4Router.Config:
        layer.feed_forward.router = derive(original, router_type, vocab_size=8, layer_id=0, n_hash_layers=1)
    else:
        layer.feed_forward.router = derive(original, router_type)
    untouched = model.layers[1].feed_forward.router

    apply_overrides(OverrideConfig(imports=[f"torchtitan_npu.override.deepseek_v4.moe_router.{factory}"]), model)

    assert type(layer.feed_forward.router) is target_type
    if router_type is not DeepSeekV41Router.Config:
        assert model.layers[1].feed_forward.router is untouched
