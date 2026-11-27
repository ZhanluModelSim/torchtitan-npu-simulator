# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 fused attention's AI_CPU records and compact-selector proof.

The model-dir entry (``models/deepseek_v4/metadata.py``) owns the row and its frames; this
module owns exactly what the AscendC kernels add to it, grouped by the consumer that launches
it -- one group per pair of kernels, one field per direction::

    aicpu_metadata
    + sparse_attn
      + fwd   -> sparse_flash_mla
      + bwd   -> sparse_flash_mla_grad
    + lightning_indexer
      + fwd   -> lightning_indexer / quant_lightning_indexer (the selector's own backend states it)
      + bwd   -> sparse_lightning_indexer_kl_loss_grad

``AicpuGeometry``
    the static numbers the AI_CPU operators are called with (heads, head dim, window, and the
    selector's widths).  It lives beside them because the context-parallel transform is handed the
    metadata alone and must rebuild it for this rank's frames; nothing in the forward or the
    backward reads it.

``AicpuMetadata``
    the AI_CPU operator outputs of one geometry, grouped by consumer, plus the geometry they were
    sized with.

``AscV4Metadata``
    the model row with this geometry's records and optional compact-selector proof attached.
    The CP backend builds its records from rank-local frames.

The compressor has no frames here: the ratio-4 overlap's carry is owned by the
context-parallel module replacement, not by the eager compressor.  The
quantized selector is another backend: it owns an entry of its own (a subclass of this one) and
rebuilds the selector group with its own contract.  Nothing here knows about quantization.
"""

from dataclasses import dataclass

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround

__all__ = [
    "AicpuGeometry",
    "AicpuMetadata",
    "AscLightningIndexerMetadata",
    "AscSparseAttentionMetadata",
    "AscV4Metadata",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class AscSparseAttentionMetadata:
    """What the fused attention's two kernels are called with: forward and backward."""

    fwd: torch.Tensor | None = None
    """``sparse_flash_mla_metadata`` output (opaque, layer-invariant)."""

    bwd: torch.Tensor | None = None
    """``sparse_flash_mla_grad_metadata`` output (opaque, layer-invariant)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AscLightningIndexerMetadata:
    """What the selector's two kernels are called with: forward and backward.

    ``fwd`` is what the selector grids its scores with -- the record of whichever selector
    backend built this entry -- and ``bwd`` what ``sparse_lightning_indexer_kl_loss_grad``
    consumes.  A layer with no compressed axis runs no selector, so both are ``None`` there.
    """

    fwd: torch.Tensor | None = None
    bwd: torch.Tensor | None = None


# The V4 selector's wire contract, shared by the fused record builder, the quantized
# backend and the converter: a TND pool of ratio-4 blocks with right-down causal masking.
LI_LAYOUT = "TND"
LI_MASK_MODE = 3
LI_CMP_RATIO = 4


def lightning_indexer_options(*, ratio: int, q: KernelFrame, cmp_k: KernelFrame) -> dict:
    """The options the selector's metadata calls and its kernels must agree on.

    They are read off this geometry's frames, so the metadata call, the LightningIndexer
    forward and the SLIKG backward cannot drift option-for-option.  ``cmp_k`` is the entry's frame
    -- on arch22 the pool this rank's reach covers, which its keys are compacted onto.
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_k=cmp_k.cu_seqlens,
        seqused_q=q.seqused,
        seqused_k=cmp_k.seqused,
        cmp_residual_k=cmp_k.residual,
        layout_q=LI_LAYOUT,
        layout_k=LI_LAYOUT,
        mask_mode=LI_MASK_MODE,
        cmp_ratio=ratio,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class AicpuGeometry:
    """The static numbers the AI_CPU operators of one geometry are called with."""

    num_heads: int
    head_dim: int
    window_size: int
    index_topk: int
    index_num_heads: int
    index_head_dim: int


@dataclass(frozen=True, kw_only=True, slots=True)
class AicpuMetadata:
    """This geometry's AI_CPU operator outputs, grouped by the kernel that launches them.

    They depend only on the frames of one forward and the layer's static geometry -- never on
    the operands -- so they are computed once per microbatch (per rank under context parallelism)
    at the eager boundary and read by every layer of the geometry instead of being re-run per
    layer inside the compiled region.  A record describes the frames it was computed for, which
    is why a re-cut frame (the CP transform's rank-local chunks) rebuilds them.
    """

    geometry: AicpuGeometry
    sparse_attn: AscSparseAttentionMetadata
    lightning_indexer: AscLightningIndexerMetadata


@dataclass(kw_only=True, slots=True)
class AscV4Metadata(DeepSeekV4Metadata):
    """The fused entry: the model row and frames, plus this geometry's AscendC records.

    The row and the frames are the model's (they describe the tensors, not the kernels); this
    entry adds what the fused kernels need on top of them:

    - ``aicpu_metadata``: the operator outputs computed for exactly these frames at the eager
      boundary and rebuilt per rank under context parallelism;
    - ``a3_workaround``: arch22's compaction of the two key streams, because every fused kernel
      right-aligns its mask at the end of the key frame it is given.

    Consumers read the fields directly (``cmp_k.seqused``, ``aicpu_metadata.sparse_attn.bwd``);
    one model runs one selection width, so there is nothing to look up.
    """

    aicpu_metadata: AicpuMetadata
    """This geometry's AI_CPU records, grouped by the kernel that reads them."""

    a3_workaround: A3KeyWorkaround | None = None
    """The two key streams as arch22 must be handed them, or ``None`` where the kernels read the
    model's own spelling (arch35, and every entry that was not built under ``if not _IS_A5:``).

    Built once per microbatch beside the records, so the forward, the backward and the selector read
    one frame and one gather each instead of re-deriving them per layer.
    """


if AscSparseAttentionMetadata not in SUPPORTED_NODES:
    register_dataclass(
        AscSparseAttentionMetadata,
        serialized_type_name=f"{AscSparseAttentionMetadata.__module__}.{AscSparseAttentionMetadata.__name__}",
    )

if AscLightningIndexerMetadata not in SUPPORTED_NODES:
    register_dataclass(
        AscLightningIndexerMetadata,
        serialized_type_name=f"{AscLightningIndexerMetadata.__module__}.{AscLightningIndexerMetadata.__name__}",
    )

if AicpuGeometry not in SUPPORTED_NODES:
    register_dataclass(AicpuGeometry, serialized_type_name=f"{AicpuGeometry.__module__}.{AicpuGeometry.__name__}")

if AicpuMetadata not in SUPPORTED_NODES:
    register_dataclass(AicpuMetadata, serialized_type_name=f"{AicpuMetadata.__module__}.{AicpuMetadata.__name__}")

if AscV4Metadata not in SUPPORTED_NODES:
    register_dataclass(AscV4Metadata, serialized_type_name=f"{AscV4Metadata.__module__}.{AscV4Metadata.__name__}")
