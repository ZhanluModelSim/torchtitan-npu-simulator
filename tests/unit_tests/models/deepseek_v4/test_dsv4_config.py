# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_debugmodel_uses_the_documented_per_layer_compression_ratios():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_spec = model_registry("debugmodel")

    assert model_spec.model.compress_ratios == (1, 1, 4, 128)
    assert len(model_spec.model.layers) == len(model_spec.model.compress_ratios)


def test_flash_mtp_4k_model_flops():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry(
        "deepseek_v4_flash",
        num_mtp_layers=1,
    ).model

    with torch.device("meta"):
        model = model_config.build()

    assert model_config.get_nparams_and_flops(model, seq_len=4096) == (
        290_942_278_866,
        92_762_352_876,
    )
