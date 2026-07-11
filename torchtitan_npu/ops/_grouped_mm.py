# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.library
import torch_npu


@torch.library.impl("aten::_grouped_mm", "PrivateUse1")
def _(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Register torch_npu.npu_grouped_matmul as the PrivateUse1 (NPU) implementation for `aten::_grouped_mm`.
    """

    # There are 3 MatMul settings in a typical MoE setting for grouped mm:
    # output = x @ w                        [n_tokens, IN] @ [n_experts, IN, OUT]
    # dx     = grad @ w.transpose(-1, -2)   [n_tokens, OUT] @ [n_experts, OUT, IN]
    # dw     = x.T @ grad                   [IN, n_tokens] @ [n_tokens, OUT]
    # We specify group_type=2 for the dw case, which is reduced along the "n_tokens" dimension,
    # a dimension that is split by the tokens that corresponds to different experts.
    # Refer to Ascend PyTorch docs:
    # https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md#参数说明
    # for explanation of group_type.
    split_along_k = self.ndim == 2 and mat2.ndim == 2

    return torch_npu.npu_grouped_matmul(
        [self],
        [mat2],
        group_list=offs.to(dtype=torch.int64),
        group_list_type=0,
        split_item=2,
        group_type=(2 if split_along_k else 0),
        bias=[bias] if bias is not None else None,
        output_dtype=out_dtype,
    )[0]
