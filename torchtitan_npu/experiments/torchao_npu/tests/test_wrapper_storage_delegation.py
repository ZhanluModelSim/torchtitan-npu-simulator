# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the wrapper untyped_storage/data_ptr delegation fix.

A storage-less wrapper tensor previously returned an "invalid python storage"
from ``untyped_storage()`` and a silent 0x0 from ``data_ptr()``, which broke
the HF safetensors save path (``storage_ptr`` / ``_to_ndarray``). These tests
assert both calls delegate to the wrapped ``_data`` tensor.

Self-contained on purpose: kept alongside the experiments package so it moves
with the package when the ao_npu test tree is relocated.
"""

import pytest
import torch
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig

from torchtitan_npu.experiments.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan_npu.experiments.torchao_npu.wrapper_tensors import (
    BaseTrainingWeightWrapperTensor,
    BlockTrainingWeightWrapperTensor,
    Float8TrainingWeightWrapperTensor,
    MXTrainingWeightWrapperTensor,
)


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockTrainingWeightWrapperTensor, BlockQuantizeConfig(), MXQuantizeConfig()),
    ],
)
def test_wrapper_untyped_storage_delegates_to_data(wrapper_cls, weight_config, act_config):
    """untyped_storage delegates to _data, exposing the real (valid) data storage."""
    w = torch.randn(64, 128)
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)

    # The wrapper holds no storage of its own; it must return _data's storage.
    assert wrapper.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
    # That storage must be a valid, non-null address (not an "invalid python storage").
    assert wrapper.untyped_storage().data_ptr() != 0


@pytest.mark.parametrize(
    "wrapper_cls, weight_config, act_config",
    [
        (BaseTrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (Float8TrainingWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
        (MXTrainingWeightWrapperTensor, MXQuantizeConfig(), MXQuantizeConfig()),
        (BlockTrainingWeightWrapperTensor, BlockQuantizeConfig(), MXQuantizeConfig()),
    ],
)
def test_wrapper_data_ptr_delegates_to_data(wrapper_cls, weight_config, act_config):
    """data_ptr delegates to _data, reporting the real data address instead of 0."""
    w = torch.randn(64, 128)
    wrapper = wrapper_cls(w, weight_config=weight_config, activation_config=act_config)

    assert wrapper.data_ptr() == w.data_ptr()
    # Regression: a storage-less wrapper used to silently return 0x0 here.
    assert wrapper.data_ptr() != 0
    # For a zero-offset contiguous tensor, data_ptr and the storage pointer agree.
    assert wrapper.data_ptr() == wrapper.untyped_storage().data_ptr()
