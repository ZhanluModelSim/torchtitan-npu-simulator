# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused RMSNorm keeps its unit-scale buffer device-resident."""

import pytest
import torch
import torch_npu

from torchtitan_npu.override.common.rms_norm import AscRMSNorm

requires_npu = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU not available"
)


@requires_npu
def test_unit_scale_buffer_lives_on_compute_device() -> None:
    norm = AscRMSNorm.Config(normalized_shape=(128,), elementwise_affine=False).build()
    norm.init_states()
    # The scale is a non-persistent buffer, so FSDP never manages it: under
    # CPU-offload construction it would stay on the host and fail the fused
    # operator's device check.
    assert norm.weight.device.type == "npu"
    assert torch.all(norm.weight == 1)
