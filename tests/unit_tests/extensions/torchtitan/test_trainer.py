# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types

import pytest

from torchtitan_npu.extensions.torchtitan import trainer as trainer_module


@pytest.mark.parametrize(
    "allow_hf32",
    [False, True],
    ids=("disabled", "enabled"),
)
def test_set_allow_hf32_updates_all_backends(monkeypatch, allow_hf32):
    fake_torch_npu = types.SimpleNamespace(
        npu=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_hf32=None),
            conv=types.SimpleNamespace(allow_hf32=None),
            aclnn=types.SimpleNamespace(allow_hf32=None),
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    trainer_module.set_allow_hf32(allow_hf32)

    assert fake_torch_npu.npu.matmul.allow_hf32 is allow_hf32
    assert fake_torch_npu.npu.conv.allow_hf32 is allow_hf32
    assert fake_torch_npu.npu.aclnn.allow_hf32 is allow_hf32
