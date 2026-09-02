# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch

from torchtitan_npu.models.deepseek_v3_2.model import Attention


def test_wkv_b_checkpoint_hooks_round_trip_plain_tensors():
    module = SimpleNamespace(
        n_heads=2,
        qk_nope_head_dim=2,
        kv_lora_rank=4,
        v_head_dim=3,
    )
    original = torch.arange(2 * (2 + 3) * 4, dtype=torch.float32).reshape(10, 4)
    state_dict = {"attention.wkv_b.weight": original.clone()}

    Attention._split_wkv_b_on_load(module, state_dict, "attention.")

    assert "attention.wkv_b.weight" not in state_dict
    assert state_dict["attention.w_uk.weight"].shape == (8, 2)
    assert state_dict["attention.w_uv.weight"].shape == (6, 4)

    Attention._merge_wkv_b_on_save(module, state_dict, "attention.", {})

    assert "attention.w_uk.weight" not in state_dict
    assert "attention.w_uv.weight" not in state_dict
    torch.testing.assert_close(state_dict["attention.wkv_b.weight"], original)
