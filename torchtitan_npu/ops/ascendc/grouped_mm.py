# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.library
import torch_npu


def _split_k_x(x: torch.Tensor) -> torch.Tensor:
    """Keep the logical shape while providing ACLNN's split-K layout.

    Autograd can pass an expanded, zero-stride gradient to the 2D x 2D
    weight-gradient path. ACLNN requires the unsplit ``x`` input of
    ``group_type=2`` to be transposed. For a logical ``[M, K]`` tensor this
    means the exact ``[1, M]`` stride pattern, so materialize a compact
    transposed copy unless the existing strides already satisfy that layout.
    """
    if x.stride(-2) == 1 and x.stride(-1) == x.size(-2):
        return x
    # ``clone()`` alone keeps the degenerate layouts of R=0/1 inputs
    # (PyTorch treats them as contiguous), so request the exact format.
    return x.transpose(-1, -2).clone(memory_format=torch.contiguous_format).transpose(-1, -2)


@torch.library.impl("aten::_grouped_mm", "PrivateUse1")
def _(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # output = x @ w                        [SEQ, IN] @ [GROUP, IN, OUT]
    # dx     = grad @ w.transpose(-1, -2)   [SEQ, OUT] @ [GROUP, OUT, IN]
    # dw     = x.T @ grad                   [IN, SEQ] @ [SEQ, OUT]
    is_dw = self.ndim == 2 and mat2.ndim == 2
    x = _split_k_x(self) if is_dw else self

    return torch_npu.npu_grouped_matmul(
        [x],
        [mat2],
        group_list=offs.to(dtype=torch.int64),
        group_list_type=0,
        split_item=2,
        # ``offs`` contains sequence-axis splits when computing ``dw``.
        group_type=2 if is_dw else 0,
        bias=bias,
        output_dtype=out_dtype,
    )[0]
