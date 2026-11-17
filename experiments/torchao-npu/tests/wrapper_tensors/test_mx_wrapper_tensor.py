# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.mx_wrapper_tensor import MXTrainingWeightWrapperTensor

from ..testing_utils import target_devices


def test_mx_handler_wraps_parameter_and_preserves_requires_grad():
    parameter = nn.Parameter(torch.randn(2, 4), requires_grad=True)
    config = ParamSwapConfig(
        weight_config=MXQuantizeConfig(),
        activation_config=MXQuantizeConfig(),
    )
    handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[MXQuantizeConfig]

    result = handler(nn.Linear(4, 2), "weight", parameter, (config,))

    assert isinstance(result.data, MXTrainingWeightWrapperTensor)
    assert result.data.weight_config is config.weight_config
    assert result.data.activation_config is config.activation_config
    assert result.requires_grad


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_keeps_the_stored_layout(device):
    """The inference weight is the stored ``[N, K]`` weight, quantized along its last axis (K)."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]

    inference_weight = MXTrainingWeightWrapperTensor(
        w, weight_config=config, activation_config=config
    ).to_inference_weight()

    assert isinstance(inference_weight, MXTensor), f"got {type(inference_weight).__name__}"
    assert inference_weight.shape == w.shape, (
        f"inference weight shape {tuple(inference_weight.shape)} != the stored {tuple(w.shape)}"
    )
    assert inference_weight.dtype is torch.bfloat16, f"logical dtype {inference_weight.dtype} != torch.bfloat16"
    assert inference_weight.quant_axis == w.ndim - 1, (
        f"quantized along dim {inference_weight.quant_axis}, expected {w.ndim - 1} (K in the stored layout)"
    )
    assert inference_weight.act_quant_config is config, "the activation config was not carried over"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_to_inference_weight_matches_the_training_forward(elem_dtype, device):
    """The inference weight has to multiply the way the wrapped one does: that is its purpose."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = MXTrainingWeightWrapperTensor(w, weight_config=config, activation_config=config)

    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, wrapped.to_inference_weight())

    assert y_inference.dtype is torch.bfloat16, f"output dtype {y_inference.dtype} != torch.bfloat16"
    assert torch.equal(y_training, y_inference), "the inference weight does not reproduce training's forward"
