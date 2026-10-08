# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.models.deepseek_v4.model import LiLoss
from torchtitan_npu.simulator import meta_env
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiLoss


def test_base_indexer_loss_is_explicitly_unsupported(monkeypatch):
    parent = LiLoss.Config(
        n_heads=2, softmax_scale=1.0, compress_ratio=4, window_size=128, layer_id=0, n_layers=1
    ).build()
    monkeypatch.setattr(meta_env, "_original_li_loss_forward", meta_env._MISSING)
    monkeypatch.setattr(LiLoss, "forward", LiLoss.forward)
    converted_forward = SimNpuLiLoss.forward
    meta_env._patch_li_loss_to_skip_buggy_einsum()
    q = torch.empty(1, 256, 2, 4, device="meta")
    with pytest.raises(ValueError, match="SimNpuLiLoss"):
        parent(q, *([None] * 10))
    assert SimNpuLiLoss.forward is converted_forward
