# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-written DeepSeek-V4 mHC Triton kernels exposed as torch custom operators.

Each kernel file registers its operators via ``torch.library.custom_op``
(with ``register_fake`` and ``register_autograd``); the differentiable mHC
compositions over these operators live in
``torchtitan_npu/override/deepseek_v4/mhc/triton.py``.
"""

__all__ = [
    "mhc_post_bmm1_op",
    "mhc_post_bmm2_op",
    "mhc_pre_bmm_op",
    "mhc_pre_only_sinkhorn_op",
    "mhc_pre_sinkhorn_op",
]

from .post_bmm1 import mhc_post_bmm1_op
from .post_bmm2 import mhc_post_bmm2_op
from .pre_bmm import mhc_pre_bmm_op
from .prepost_sinkhorn import mhc_pre_only_sinkhorn_op, mhc_pre_sinkhorn_op
