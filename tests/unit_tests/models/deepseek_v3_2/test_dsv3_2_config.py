# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial


def test_debugmodel_indexer_weight_projection_uses_linear_initialization():
    from torchtitan_npu.models.deepseek_v3_2 import model_registry

    model_spec = model_registry("debugmodel")
    init = model_spec.model.layers[0].attention.indexer.weights_proj.param_init

    assert isinstance(init["weight"], partial)
    assert init["weight"].func.__name__ == "trunc_normal_"
    assert init["weight"].keywords == {"std": 0.02}
