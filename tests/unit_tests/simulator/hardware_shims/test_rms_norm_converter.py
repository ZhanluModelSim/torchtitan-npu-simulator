# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.converters.kernels.rms_norm import RMSNormModelConfig
from torchtitan_npu.simulator.hardware_shims.rms_norm_converter import (
    apply_rms_norm_shims,
    unapply_rms_norm_shims,
)
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import SimRMSNorm


class _FakeModelSpec:
    name = "deepseek_v4"


def test_apply_rms_norm_shims_replaces_and_restores_converter():
    original = RMSNormModelConfig.model_converter
    try:
        apply_rms_norm_shims()
        assert RMSNormModelConfig.model_converter is not original
    finally:
        unapply_rms_norm_shims()
        assert RMSNormModelConfig.model_converter is original


def test_sim_rms_norm_converter_replaces_rms_norm_modules():
    apply_rms_norm_shims()
    try:
        model = nn.Sequential(RMSNorm(RMSNorm.Config(normalized_shape=16, eps=1e-6)))
        converter = RMSNormModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model[0], SimRMSNorm)
    finally:
        unapply_rms_norm_shims()


def test_unapply_rms_norm_shims_is_idempotent():
    unapply_rms_norm_shims()
    unapply_rms_norm_shims()
