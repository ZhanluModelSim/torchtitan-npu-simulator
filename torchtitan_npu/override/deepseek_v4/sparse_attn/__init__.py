# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4 sparse attention using CANN SMLA and its backward."""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override
from torchtitan.models.deepseek_v4.attention import DSV4InnerAttention

if TYPE_CHECKING:
    from .ascendc import AscCompressedSparseInnerAttention


@override(target=DSV4InnerAttention.Config, description="Use CANN SMLA for DeepSeek V4 sparse attention")
def asc(
    cfg: DSV4InnerAttention.Config,
    *,
    num_heads: int = 64,
    head_dim: int = 512,
    index_num_heads: int = 64,
    index_head_dim: int = 128,
) -> "AscCompressedSparseInnerAttention.Config":
    """Select the attention kernel. The indexer backend is configured on indexer.inner_indexer."""
    from .ascendc import AscCompressedSparseInnerAttention

    return derive(
        cfg,
        AscCompressedSparseInnerAttention.Config,
        num_heads=num_heads,
        head_dim=head_dim,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )
