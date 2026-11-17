# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HiFloat8 quantization primitives."""

__all__ = ["quantize_hifloat8"]

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig


def quantize_hifloat8(tensor: torch.Tensor, config: HiF8QuantizeConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to HiFloat8 with a float32 scale. Requires
    ``quant_mode == "pertensor"``: its single scalar scale is permutation-
    invariant, so quantizing on a dense (descending-stride) view instead of
    whatever layout the tensor arrived in avoids the implicit
    ``Contiguous``/``Transpose`` copy ``npu_dynamic_quant`` does internally
    for a non-dense input -- without an ``is_contiguous()`` check, which
    would evaluate as a guard under ``torch.compile``. ``pertoken``/
    ``perchannel`` reduce over a specific axis, which permuting would
    silently change, hence the check below.
    """
    if config.quant_mode != "pertensor":
        raise ValueError(f"quantize_hifloat8 requires quant_mode='pertensor', got {config.quant_mode!r}.")

    if tensor.dtype not in (torch.float16, torch.bfloat16):
        tensor = tensor.to(torch.bfloat16)

    perm_indices = sorted(range(tensor.ndim), key=lambda d: tensor.stride(d), reverse=True)
    tensor_p = tensor.permute(perm_indices)

    y_p, scale = torch_npu.npu_dynamic_quant(
        tensor_p, dst_type=config.elem_dtype, dst_type_max=config.dst_type_max, quant_mode=config.quant_mode
    )

    # Inverse permutation: y_p dim j corresponds to tensor dim perm_indices[j].
    perm_back_indices = [0] * tensor.ndim
    for j, p in enumerate(perm_indices):
        perm_back_indices[p] = j
    y = y_p.permute(perm_back_indices)

    return y, scale.to(torch.float32)
