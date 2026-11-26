# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration-only V4.1 recipes with test fixture resources bound in Python."""

import os

from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model
from torchtitan_npu_recipes.tests.models import deepseek_v4_1 as test_recipes


def deepseek_v4_1_debugmodel_text():
    return test_recipes._make_trainer_config(
        "deepseek_v4_1_debugmodel_text",
        hf_assets_path="tests/assets/deepseek_v3",
    )


def deepseek_v4_1_debugmodel_multimodal():
    config = test_recipes._make_trainer_config(
        "deepseek_v4_1_debugmodel",
        hf_assets_path="tests/assets/deepseek_v3",
    )
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.dataloader = test_recipes._vision_dataloader_config(
        config.model,
        dataset="cc12m-test",
        dataset_path="tests/assets/cc12m_test",
        num_packing_bins=4,
    )
    return config


def deepseek_v4_1_debugmodel_multimodal_sft():
    return test_recipes.deepseek_v4_1_debugmodel_multimodal_sft(
        dataset_path=os.environ["DSV41_SFT_FIXTURE_PATH"],
    )
