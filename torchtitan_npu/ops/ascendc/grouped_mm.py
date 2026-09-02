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
    # output = x @ w                        [SEQ, IN] @ [GROUP, IN, OUT]
    # dx     = grad @ w.transpose(-1, -2)   [SEQ, OUT] @ [GROUP, OUT, IN]
    # dw     = x.T @ grad                   [IN, SEQ] @ [SEQ, OUT]
    is_dw = self.ndim == 2 and mat2.ndim == 2

    return torch_npu.npu_grouped_matmul(
        [self],
        [mat2],
        group_list=offs.to(dtype=torch.int64),
        group_list_type=0,
        split_item=2,
        # ``offs`` contains sequence-axis splits when computing ``dw``.
        group_type=2 if is_dw else 0,
        bias=bias,
        output_dtype=out_dtype,
    )[0]
