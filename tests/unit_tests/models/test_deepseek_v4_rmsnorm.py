# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.converters.kernels.rms_norm import (
    NPURMSNorm,
    NpuRMSNormConverter,
)
from torchtitan_npu.models.deepseek_v4 import deepseekv4_configs


def test_deepseek_v4_rmsnorm_converter_replaces_every_norm():
    model = deepseekv4_configs["smoketest"]().build()
    expected_names = {
        name for name, module in model.named_modules() if isinstance(module, RMSNorm)
    }

    assert expected_names

    converter = NpuRMSNormConverter(
        SimpleNamespace(name="deepseek_v4"),
    )
    converter.convert(model)

    converted_names = {
        name for name, module in model.named_modules() if isinstance(module, NPURMSNorm)
    }
    assert converted_names == expected_names
