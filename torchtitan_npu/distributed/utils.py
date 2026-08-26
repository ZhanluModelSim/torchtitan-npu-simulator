# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def set_allow_hf32(allow: bool) -> None:
    """Set HF32 consistently on the NPU math backends."""

    import torch_npu

    torch_npu.npu.matmul.allow_hf32 = allow
    torch_npu.npu.conv.allow_hf32 = allow
    torch_npu.npu.aclnn.allow_hf32 = allow
