# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch


def dequantize_mxfp8_on_npu(
    qdata: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    crop_start: int,
    requested_length: int,
    target: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    import torch_npu

    qdata_host = qdata if qdata.is_pinned() else qdata.pin_memory()
    scale_host = scale_u8 if scale_u8.is_pinned() else scale_u8.pin_memory()
    qdata_npu = qdata_host.to(target.device, non_blocking=True)
    paired_scale_host = scale_host.view(*scale_host.shape[:-1], -1, 2)
    scale_npu = paired_scale_host.to(target.device, non_blocking=True)
    aligned_fp32 = torch_npu.npu_anti_mx_quant(
        qdata_npu,
        scale_npu,
        axis=-1,
        dst_type=torch.float32,
        src_type=torch.float8_e4m3fn,  # pyrefly: ignore [missing-attribute]
    )
    target.copy_(aligned_fp32.narrow(-1, crop_start, requested_length))
    return qdata_host, scale_host, paired_scale_host, qdata_npu, scale_npu, scale_npu, aligned_fp32
