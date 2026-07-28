# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: use scaled dot-product attention on NPU."""

from torchtitan.config import override
from torchtitan.models.common.attention import FlexAttention, ScaledDotProductAttention


@override(
    target=FlexAttention.Config,
    exact=True,
    description="Replace FlexAttention with ScaledDotProductAttention for NPU compatibility",
)
def npu_sdpa_override(cfg: FlexAttention.Config) -> ScaledDotProductAttention.Config:
    return ScaledDotProductAttention.Config()
