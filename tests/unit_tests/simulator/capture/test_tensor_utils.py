# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.tensor_utils import (
    dtype_byte_size,
    dtype_to_str,
    tensor_volume_bytes,
    to_tensor_meta,
)


def test_dtype_to_str_known_dtypes():
    assert dtype_to_str(torch.float32) == "float32"
    assert dtype_to_str(torch.bfloat16) == "bfloat16"
    assert dtype_to_str(torch.int64) == "int64"


def test_dtype_byte_size_known_and_unknown():
    assert dtype_byte_size("float32") == 4
    assert dtype_byte_size("bfloat16") == 2
    assert dtype_byte_size("int64") == 8
    assert dtype_byte_size("totally_unknown_dtype") == 4  # graceful fallback


def test_tensor_volume_bytes_computes_correctly():
    assert tensor_volume_bytes((2, 3, 4), "float32") == 2 * 3 * 4 * 4
    assert tensor_volume_bytes((10,), "bfloat16") == 20


def test_to_tensor_meta_from_meta_tensor():
    t = torch.empty(2, 3, dtype=torch.bfloat16, device="meta")
    meta = to_tensor_meta(t, name="x")
    assert meta.name == "x"
    assert meta.shape == (2, 3)
    assert meta.dtype == "bfloat16"
    assert meta.device == "meta"
    assert meta.is_parameter is False


def test_to_tensor_meta_marks_parameter():
    t = torch.empty(4, 4, device="meta")
    meta = to_tensor_meta(t, name="w", is_parameter=True)
    assert meta.is_parameter is True
