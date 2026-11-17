# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def dequantize_mx_on_npu(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    crop_start: int,
    requested_length: int,
    target: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Dequantize packed E2M1 or E4M3 weights with E8M0 block scales."""
    import torch_npu

    if qdata.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise ValueError(f"Unsupported MX weight dtype: {qdata.dtype}")
    if not target.is_floating_point():
        raise ValueError("MX loading requires a floating-point target")
    # TODO: Select the quantization format from checkpoint config when available.
    src_type = torch_npu.float4_e2m1fn_x2 if qdata.dtype == torch.uint8 else torch.float8_e4m3fn
    scale_bytes = scale.contiguous().view(torch.uint8)
    if scale_bytes.shape[-1] % 2:
        scale_bytes = torch.cat((scale_bytes, torch.full_like(scale_bytes[..., :1], 127)), dim=-1)
    qdata_host = qdata.contiguous().pin_memory()
    scale_host = scale_bytes.pin_memory()
    paired_scale = scale_host.reshape(*scale_host.shape[:-1], -1, 2)
    qdata_npu = qdata_host.to(target.device, non_blocking=True)
    scale_npu = paired_scale.to(target.device, non_blocking=True)
    aligned_fp32 = torch_npu.npu_anti_mx_quant(
        qdata_npu,
        scale_npu,
        axis=-1,
        dst_type=torch.float32,
        src_type=src_type,
    )
    target.copy_(aligned_fp32.narrow(-1, crop_start, requested_length))
    return qdata_host, scale_host, paired_scale, qdata_npu, scale_npu, aligned_fp32
