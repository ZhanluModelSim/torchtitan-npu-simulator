# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    deepseek_v4_1_debugmodel_multimodal,
    deepseek_v4_1_debugmodel_text,
)


@pytest.mark.parametrize(
    "factory", (deepseek_v4_1_debugmodel_text, deepseek_v4_1_debugmodel_multimodal), ids=("text", "multimodal")
)
def test_debug_recipe_builds_with_token_budget_and_preserves_embedding_weights(factory):
    config = factory()
    config.model.update_from_config(config=config)

    with torch.device("meta"):
        model = config.model.build()

    assert config.training.num_tokens_per_microbatch_per_dp_rank == 512
    assert config.training.max_context_length == config.model.max_context_length == 512
    assert tuple(model.tok_embeddings.weight.shape) == (129280, 512)
    assert tuple(model.lm_head.weight.shape) == (129280, 512)
    assert len(model.layers) == 40
