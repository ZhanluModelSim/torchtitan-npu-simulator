# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadFP32OutputConverter
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear

from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "flavor,layers,experts,context",
    [
        ("debugmodel", 4, 4, 16384),
        ("deepseek_v4_flash", 43, 256, 4096),
        ("deepseek_v4_flash_43layers_16experts", 43, 16, 4096),
        ("deepseek_v4_pro", 61, 384, 4096),
        ("deepseek_v4_pro_61layers_32experts", 61, 32, 4096),
        ("deepseek_v4_pro_61layers_12experts", 61, 12, 4096),
    ],
    ids=["debug", "flash", "flash-reduced", "pro", "pro-32", "pro-12"],
)
def test_flavor_preserves_model_geometry_and_context_limit(flavor, layers, experts, context):
    model = build_model_config(flavor, parallelism=ParallelismConfig())

    assert len(model.layers) == model.n_layers == layers
    assert model.layers[-1].feed_forward.num_experts == experts
    assert model.max_context_length == context
    with pytest.raises(ValueError, match="seq_len"):
        build_model_config(flavor, parallelism=ParallelismConfig(), seq_len=context + 1)


@pytest.mark.parametrize("mtp_depth", [0, 1], ids=["no-mtp", "mtp"])
def test_debug_converter_keeps_mtp_loss_and_data_contract(mtp_depth):
    config = deepseek_v4_debugmodel(
        num_mtp_layers=mtp_depth,
        converters=[LMHeadFP32OutputConverter.Config()],
    )

    assert isinstance(config.model.lm_head, HiMidLoLinear.Config)
    assert len(config.model.mtp_layers or ()) == mtp_depth
    assert config.dataloader.num_mtp_layers == mtp_depth
    assert isinstance(config.loss, MTPChunkedLossWrapper.Config) is bool(mtp_depth)
