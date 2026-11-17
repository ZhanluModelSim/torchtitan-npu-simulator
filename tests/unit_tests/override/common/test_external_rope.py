# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.override.common.rope import AscHalfRotation


def test_fused_half_adapter_preserves_fp32_tables_and_batches(monkeypatch):
    import torchtitan_npu.override.common.rope as fused

    calls = []

    def kernel(x, cos, sin, *, rotary_mode):
        calls.append((x, cos, sin, rotary_mode))
        return x + 2

    monkeypatch.setattr(fused.torch_npu, "npu_rotary_mul", kernel)
    x = torch.ones(2, 3, 1, 4, dtype=torch.bfloat16)
    cos = torch.linspace(0.1, 0.9, 2 * 3 * 2).reshape(2, 3, 1, 2)
    sin = cos.flip(1)
    actual = AscHalfRotation.Config().build()(x, cos, sin, inverse=True)
    value, sent_cos, sent_sin, rotary_mode = calls[0]
    assert value.dtype == sent_cos.dtype == sent_sin.dtype == torch.float32
    assert sent_cos.shape[0] == 1
    expected_cos = torch.cat((cos, cos), -1)
    expected_sin = -torch.cat((sin, sin), -1)
    # Distinct per-image tables fold batch into the shared axis.
    expected_cos = expected_cos.flatten(0, 1).unsqueeze(0)
    expected_sin = expected_sin.flatten(0, 1).unsqueeze(0)
    torch.testing.assert_close(sent_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sent_sin, expected_sin, rtol=0, atol=0)
    assert rotary_mode == "half"
    torch.testing.assert_close(actual, torch.full_like(x, 3), rtol=0, atol=0)
