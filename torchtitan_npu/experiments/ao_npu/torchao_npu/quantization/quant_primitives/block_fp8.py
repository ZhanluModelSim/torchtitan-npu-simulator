# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block FP8 low-level primitives for NPU.

These are pure tensor helpers (no autograd, no matmul) used by higher-level
ops in :mod:`torchtitan_npu.experiments.ao_npu.torchao_npu.ops.block_ops`
(e.g., ``_BlockFP8QuantMM.forward`` calls :func:`quantize_right_operand`).
"""

import torch
import torch_npu

from ..quant_configs import BlockQuantizeConfig


def quantize_right_operand(
    tensor: torch.Tensor,
    axis: int,
    config: BlockQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply mxfp4 fake-quant (optional) then block FP8 quantize the right operand of matmul.

    Returns the quantized tensor and the two scale tensors produced by
    ``npu_dynamic_block_mx_quant``.
    """
    if config.mxfp4_fake_quantize_config is not None:
        # Lazy import to avoid circular dependency: ops.mx_ops imports
        # quant_primitives.mx (for mxfp4_dequantize), and quant_primitives.block_fp8
        # imports ops.mx_ops (for mxfp4_fake_quantize). Loading mxfp4_fake_quantize
        # lazily breaks the cycle.
        from ...ops.mx_ops import mxfp4_fake_quantize

        tensor = mxfp4_fake_quantize(tensor, config.mxfp4_fake_quantize_config, axis=axis)
    return torch_npu.npu_dynamic_block_mx_quant(
        tensor,
        dst_type=config.elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
