# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 LightningIndexer override: selection fused with its SLIKG backward.

This entry and :mod:`~torchtitan_npu.override.deepseek_v4_1.sparse_attn` have separate
targets so TorchAO-NPU can independently replace either BF16 fallback.  The resulting
configuration must still contain both a teacher provider and a consumer, and both sides
must use document-local indices.

This override drives only the BF16 ``lightning_indexer`` forward. The ds41 QLI/QSLI
quantized forward bridge is isolated in the optional ``torchao-npu`` package under
``experiments/torchao-npu`` and is not selected by this override.
"""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.indexer import InnerIndexer

if TYPE_CHECKING:
    from .ascendc import AscInnerIndexer


@override(
    target=InnerIndexer.Config,
    exact=True,
    description="BF16 LightningIndexer selection fused with its SLIKG backward in one autograd.Function",
)
def asc(cfg: InnerIndexer.Config) -> "AscInnerIndexer.Config":
    from .ascendc import AscInnerIndexer

    return derive(cfg, AscInnerIndexer.Config)
