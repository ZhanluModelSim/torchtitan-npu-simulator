# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the torchao safetensors allowlist registration."""

from torchao.prototype.safetensors import safetensors_utils
from torchao_npu.quantized_tensors import MXTensor
from torchao_npu.serialization import register_torchao_safetensors


def test_register_adds_mx_tensor_and_configs_to_allowlists():
    # Importing torchao_npu.serialization already ran the registration;
    # this asserts what the registration installed.
    # Identity, not name: the upstream CUDA MXTensor shares the name, so only
    # ``is`` proves the deliberate overwrite took effect.
    assert safetensors_utils.ALLOWED_CLASSES["MXTensor"] is MXTensor
    assert safetensors_utils.ALLOWED_CLASSES["MXQuantizeConfig"].__name__ == "MXQuantizeConfig"
    assert safetensors_utils.ALLOWED_CLASSES["BlockMXQuantizeConfig"].__name__ == "BlockMXQuantizeConfig"
    assert "MXTensor" in safetensors_utils.ALLOWED_TENSORS_SUBCLASSES


def test_register_is_idempotent():
    register_torchao_safetensors()
    register_torchao_safetensors()

    assert safetensors_utils.ALLOWED_TENSORS_SUBCLASSES.count("MXTensor") == 1
