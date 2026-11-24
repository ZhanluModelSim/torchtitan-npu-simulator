# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The quantized selector's record for the DeepSeek-V4 fused attention.

Only one record of the entry depends on precision: the selector's forward grid.  The
sliding-window pair and SLIG's backward record are the same operators with the same arguments
in both paths, so this module buys exactly one extra operator call
(``quant_lightning_indexer_metadata``) on top of the fused backend's builder, and replaces the
selector field of the entry it produced.

The operator lives in ``cann_ops_transformer`` -- the same package as the BF16 record operator,
with the same frames and one extra ``quant_mode`` integer -- so the fused backend never needs to
know that a quantized selector exists: this backend states the ABI on its own config and builds
its own entry.
"""

from __future__ import annotations

__all__ = [
    "CMP_RATIO",
    "LAYOUT",
    "MASK_MODE",
    "QuantV4Metadata",
    "attach_selector_record",
    "selector_record",
]

from dataclasses import dataclass, replace

import cann_ops_transformer  # noqa: F401  # register torch.ops.cann_ops_transformer
import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_CMP_RATIO as CMP_RATIO,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_LAYOUT as LAYOUT,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_MASK_MODE as MASK_MODE,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    AicpuGeometry,
    AscV4Metadata,
)


@dataclass(kw_only=True, slots=True)
class QuantV4Metadata(AscV4Metadata):
    """The quantized backend's entry.

    Same interface as the fused one (``aicpu_metadata.sparse_attn`` /
    ``aicpu_metadata.lightning_indexer``), because the forward it feeds is the fused forward.
    A context-parallel re-cut is refused: this entry's selector record is this backend's own
    operator, and the re-cut has no backend to ask.
    """

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        attention_metadata: QuantV4Metadata,
        *,
        permutation: torch.Tensor | None,
    ) -> QuantV4Metadata:
        del attention_metadata, permutation
        raise ValueError(
            "the quantized DeepSeek-V4 selector does not rebuild its record per rank yet: run "
            "with --parallelism.context-parallel-degree 1."
        )


if QuantV4Metadata not in SUPPORTED_NODES:
    register_dataclass(QuantV4Metadata, serialized_type_name=f"{QuantV4Metadata.__module__}.{QuantV4Metadata.__name__}")


def selector_record(
    metadata: AscV4Metadata,
    *,
    geometry: AicpuGeometry,
    quant_mode: int,
) -> torch.Tensor:
    """The quantized selector's forward grid for this entry's frames."""
    cmp_k = metadata.cmp_k
    if cmp_k is None or cmp_k.residual is None:
        raise ValueError("the quantized selector needs a pooling geometry's compressed frame.")
    return torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        geometry.index_num_heads,
        1,
        geometry.index_head_dim,
        geometry.index_topk,
        quant_mode,
        layout_q=LAYOUT,
        layout_k=LAYOUT,
        mask_mode=MASK_MODE,
        cmp_ratio=CMP_RATIO,
        cu_seqlens_q=metadata.q.cu_seqlens,
        cu_seqlens_k=cmp_k.cu_seqlens,
        cmp_residual_k=cmp_k.residual,
    )


def attach_selector_record(
    metadata: AscV4Metadata,
    *,
    geometry: AicpuGeometry,
    quant_mode: int,
) -> AscV4Metadata:
    """Replace the entry's selector record with the quantized one.

    Only the geometry that runs the selector has one: every other ratio keeps the fused entry
    untouched (and rebuildable by the fused context-parallel transform).  The fused builder's
    other records are kept: they are the same operators either path launches.  The replaced
    tensor carries the same dynamic-shape annotations as the record it replaces, so a new
    batch's block count does not specialize the graph.
    """
    if metadata.ratio != CMP_RATIO:
        return metadata
    record = selector_record(metadata, geometry=geometry, quant_mode=quant_mode)
    torch._dynamo.maybe_mark_dynamic(record, 0)
    # GraphTrainer's tracer reads this annotation for input dimensions.
    setattr(record, "_dynamo_unbacked_indices", {0})  # noqa: B010
    asc = replace(
        metadata.aicpu_metadata,
        lightning_indexer=replace(metadata.aicpu_metadata.lightning_indexer, fwd=record),
    )
    return QuantV4Metadata(
        ratio=metadata.ratio,
        documents=metadata.documents,
        position_k=metadata.position_k,
        q=metadata.q,
        swa_k=metadata.swa_k,
        cmp_k=metadata.cmp_k,
        aicpu_metadata=asc,
        a3_workaround=metadata.a3_workaround,
    )
