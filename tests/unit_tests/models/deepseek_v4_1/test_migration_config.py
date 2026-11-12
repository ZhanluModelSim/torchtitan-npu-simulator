# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu_recipes.models.deepseek_v4_1 import (
    deepseek_v4_1_debugmodel_multimodal,
    deepseek_v4_1_debugmodel_text,
)
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41MultimodalModel


@pytest.mark.parametrize(
    "factory,expected_multimodal",
    ((deepseek_v4_1_debugmodel_text, False), (deepseek_v4_1_debugmodel_multimodal, True)),
    ids=("text", "multimodal"),
)
def test_debug_recipe_builds_with_token_budget_and_selective_fp32_parameters(factory, expected_multimodal):
    config = factory()
    config.hf_assets_path = "tests/assets/deepseek_v3"
    config.model.update_from_config(config=config)

    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = config.model.build()

    assert config.training.num_tokens_per_microbatch_per_dp_rank == 512
    assert config.training.max_context_length == config.model.max_context_length == 512
    assert tuple(model.tok_embeddings.weight.shape) == (129280, 512)
    assert tuple(model.lm_head.weight.shape) == (129280, 512)
    assert len(model.layers) == 40
    assert isinstance(model, DeepSeekV41MultimodalModel) is expected_multimodal
    assert model.config.fp32_param_patterns == config.model.fp32_param_patterns
    assert config.model.fp32_param_patterns == (
        "layers.*.hc_attn_pre.*",
        "layers.*.hc_ffn_pre.*",
        "layers.*.attention.attn_sink",
    )
    assert config.training.mixed_precision_param == "bfloat16"
    assert config.training.mixed_precision_reduce == "float32"
    assert config.training.extension.allow_hf32 is True
