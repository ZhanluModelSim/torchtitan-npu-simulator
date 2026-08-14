# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.smoke_tests.conftest import skip_on_runtime_unsupported

pytestmark = pytest.mark.smoke


def test_quant_linear_mxfp8(npu_device):
    import torch_npu

    x = torch.randn(16, 256, dtype=torch.bfloat16, device=npu_device)
    weight = torch.randn(512, 256, dtype=torch.bfloat16, device=npu_device)

    try:
        x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1)
        weight_quant, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
    except RuntimeError as error:
        skip_on_runtime_unsupported(
            error,
            (
                "does not support opType [DynamicMxQuant]",
                "SoC version ascend910b verification failed",
            ),
            "DynamicMxQuant is not supported on the current Ascend SOC",
        )

    assert x_quant.shape == x.shape
    assert weight_quant.shape == weight.shape
    assert x_scale is not None
    assert weight_scale is not None
