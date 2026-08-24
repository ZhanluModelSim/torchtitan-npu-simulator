# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block FP8 low-level primitives for NPU.

These are pure tensor helpers (no autograd, no matmul) used by higher-level
ops in :mod:`torchao_npu.ops.block_ops`
(e.g., ``_BlockFP8QuantMM.forward`` calls :func:`block_fp8_quantize`).
"""

import importlib

import torch
import torch_npu

from ..quant_configs import BlockQuantizeConfig

importlib.import_module("cann_ops_nn.ops")


def block_fp8_quantize(
    tensor: torch.Tensor,
    axis: int,
    config: BlockQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply mxfp4 fake-quant (optional) then block FP8 quantize the right operand of matmul.

    When ``mxfp4_fake_quantize_config`` is set, the MXFP4 quantized tensor is
    converted directly to block FP8 via the fused kernel
    ``npu_mx_to_block_mx_quant`` (with a transpose workaround since the fused
    kernel requires packing along the last dimension).

    Returns the quantized tensor and the two scale tensors produced by
    ``npu_dynamic_block_mx_quant`` or the fused kernel.
    """
    if config.mxfp4_fake_quantize_config is not None:
        fq_cfg = config.mxfp4_fake_quantize_config

        if axis == -1:
            # MXFP4 quant + fused op directly (packing already on last dim)
            fp4_tensor, mxscale = torch_npu.npu_dynamic_mx_quant(
                tensor,
                axis=-1,
                dst_type=torch_npu.float4_e2m1fn_x2,
                block_size=fq_cfg.block_size,
                round_mode=fq_cfg.round_mode,
                scale_alg=fq_cfg.scale_alg,
                dst_type_max=fq_cfg.dst_type_max,
            )
            return torch.ops.cann_ops_nn.mx_to_block_mx_quant.default(
                fp4_tensor,
                mxscale,
                dst_type=config.elem_dtype,
                x_type=torch_npu.float4_e2m1fn_x2,
            )

        elif axis == -2:
            # Transpose so K-dim becomes last, fused op, then swap + transpose back
            tensor_p = tensor.transpose(-2, -1)

            fp4_tensor, mxscale = torch_npu.npu_dynamic_mx_quant(
                tensor_p,
                axis=-1,
                dst_type=torch_npu.float4_e2m1fn_x2,
                block_size=fq_cfg.block_size,
                round_mode=fq_cfg.round_mode,
                scale_alg=fq_cfg.scale_alg,
                dst_type_max=fq_cfg.dst_type_max,
            )

            B_q_p, B_s1_p, B_s2_p = torch.ops.cann_ops_nn.mx_to_block_mx_quant.default(
                fp4_tensor,
                mxscale,
                dst_type=config.elem_dtype,
                x_type=torch_npu.float4_e2m1fn_x2,
            )

            B_q = B_q_p.transpose(-2, -1)
            B_s1 = B_s2_p.transpose(-3, -2)
            B_s2 = B_s1_p.transpose(-3, -2)
            return B_q, B_s1, B_s2

        else:
            raise ValueError(f"Only axis=-1 and axis=-2 are supported, got axis={axis}")

    # No mxfp4: direct block FP8 quant
    return torch_npu.npu_dynamic_block_mx_quant(
        tensor,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
