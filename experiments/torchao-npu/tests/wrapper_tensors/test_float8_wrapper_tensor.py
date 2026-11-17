# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.wrapper_tensors.float8_wrapper_tensor import Float8TrainingWeightWrapperTensor


def test_float8_handler_wraps_parameter_and_preserves_requires_grad():
    parameter = nn.Parameter(torch.randn(2, 4), requires_grad=True)
    config = ParamSwapConfig(weight_config=Float8FakeQuantizeConfig())
    handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[Float8FakeQuantizeConfig]

    result = handler(nn.Linear(4, 2), "weight", parameter, (config,))

    assert isinstance(result.data, Float8TrainingWeightWrapperTensor)
    assert result.data.weight_config is config.weight_config
    assert result.requires_grad


def test_to_inference_weight_is_not_implemented():
    """Float8 has no inference-side quantized tensor yet, so the inherited base method refuses."""
    wrapper = Float8TrainingWeightWrapperTensor(torch.randn(2, 4), weight_config=Float8FakeQuantizeConfig())

    with pytest.raises(NotImplementedError, match="to_inference_weight"):
        wrapper.to_inference_weight()
