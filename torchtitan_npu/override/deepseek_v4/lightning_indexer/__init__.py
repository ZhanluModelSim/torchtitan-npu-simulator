# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 InnerIndexer override: selection fused with its SLIKG backward.

The selector and the sparse attention are separate targets, exactly as in the V4.1 stack:
this entry emits ``topk_indices`` (and the teacher carrier ``topk_scores``) and consumes the
teacher on that carrier's gradient; ``sparse_attn.asc`` emits the teacher from SMLAG's
backward. Three ports of the same pair are registered here: ``asc`` binds the CANN kernels,
``pypto`` the PyPTO drop-ins, and ``pypto_pro`` the PyPTO-Pro kernels. A training recipe selects
one selector backend together with ``sparse_attn.asc``.
"""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer

if TYPE_CHECKING:
    from .ascendc import AscInnerIndexer
    from .pypto import PyPTOInnerIndexer
    from .pypto_pro import PyPTOProInnerIndexer


@override(
    target=InnerIndexer.Config,
    exact=True,
    description="BF16 InnerIndexer selection fused with its SLIKG backward in one autograd.Function",
)
def asc(cfg: InnerIndexer.Config) -> "AscInnerIndexer.Config":
    from .ascendc import AscInnerIndexer

    return derive(cfg, AscInnerIndexer.Config)


@override(
    target=InnerIndexer.Config,
    exact=True,
    description="PyPTO InnerIndexer selection fused with its SLIKG backward in one autograd.Function",
)
def pypto(cfg: InnerIndexer.Config) -> "PyPTOInnerIndexer.Config":
    from .pypto import PyPTOInnerIndexer

    return derive(cfg, PyPTOInnerIndexer.Config)


@override(
    target=InnerIndexer.Config,
    exact=True,
    description="PyPTO-Pro InnerIndexer selection fused with its SLIKG backward in one autograd.Function",
)
def pypto_pro(cfg: InnerIndexer.Config) -> "PyPTOProInnerIndexer.Config":
    from .pypto_pro import PyPTOProInnerIndexer

    return derive(cfg, PyPTOProInnerIndexer.Config)
