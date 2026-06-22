# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

pytest.importorskip("torchtitan.models.deepseek_v3")

from torchtitan_npu.models.deepseek_v4.model.state_dict_adapter import (  # noqa: E402
    DeepSeekV4StateDictAdapter,
)


def _make_adapter_with_captured_to_hf_new():
    adapter = object.__new__(DeepSeekV4StateDictAdapter)
    adapter._input_format = "hf"
    captured = {}

    def fake_to_hf_new(state_dict):
        captured["state_dict"] = state_dict
        return state_dict

    adapter.to_hf_new = fake_to_hf_new
    return adapter, captured


def test_deepseek_v4_to_hf_splits_w13_before_hf_mapping():
    adapter, captured = _make_adapter_with_captured_to_hf_new()
    w13 = torch.arange(2 * 8 * 8, dtype=torch.bfloat16).reshape(2, 8, 8)
    other_weight = torch.ones(2, 2, dtype=torch.bfloat16)

    result = adapter.to_hf(
        {
            "layers.0.moe.experts.w13": w13,
            "layers.0.attention.weight": other_weight,
        }
    )

    assert result is captured["state_dict"]
    assert "layers.0.moe.experts.w13" not in result
    assert torch.equal(result["layers.0.moe.experts.w1"], w13[:, :4, :])
    assert torch.equal(result["layers.0.moe.experts.w3"], w13[:, 4:, :])
    assert result["layers.0.moe.experts.w1"].dtype == torch.bfloat16
    assert result["layers.0.moe.experts.w3"].dtype == torch.bfloat16
    assert result["layers.0.attention.weight"] is other_weight
