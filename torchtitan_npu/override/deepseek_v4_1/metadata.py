# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 AscendC attention's metadata hierarchy.

Each level is named for what it is, and the AI_CPU metadata is grouped by the consumer that
launches it -- one group per pair of kernels, one field per direction::

    aicpu_metadata
    + sparse_attn
      + fwd   -> sparse_flash_mla
      + bwd   -> sparse_flash_mla_grad
    + lightning_indexer
      + fwd   -> lightning_indexer
      + bwd   -> sparse_lightning_indexer_kl_loss_grad

The selector's group carries two fields more than the BF16 selector fills: ``fwd_sparse`` (the
sparse search's schedule) and ``sparse_length`` (the per-row candidate count it is bound by).  They
stay ``None`` on this stack, whose kernels score or publish a pool but whose search's schedule
arrives with the non-quantized candidate operators; they are stated here so that one shape names
both paths.  The quantized attention is a *backend of its own*, not an extension of this
hierarchy: it builds its own entry, of its own classes, in its own package.

``AicpuGeometry``
    the static numbers the AI_CPU operators are called with (heads, head dim, window, and the
    indexer's widths and selection width).  It lives beside them because the context-parallel
    transform is handed the metadata alone and must rebuild it for this rank's frames; nothing in
    the forward or the backward reads it.

``AicpuMetadata``
    the AI_CPU operator outputs of one geometry, grouped by consumer, plus the geometry they were
    sized with.  One model runs one selection width, so each output is one field rather than a
    table.

``KernelFrame``
    one tensor's coordinate frame: its ragged document boundaries, the per-document prefix this
    consumer addresses, and the trailing partial group of a compressed axis.  An entry is three of
    them (queries, sliding-window keys, compressed keys) plus the row.

Consumers read the fields: ``attention_metadata.cmp_k.seqused``,
``attention_metadata.aicpu_metadata.sparse_attn.bwd``.  There are no lookup helpers: the entry
*is* one geometry and the model runs one selection width, so there is nothing to look up.  The
geometry is *not* cross-checked against the layer's own config either -- the override is the only
place that states it, and the kernels are where a wrong statement surfaces.

Nothing here is quantization: this module states only what the AscendC attention's and the BF16
selector's own kernels are called with.  A stack that runs other kernels (the quantized attention)
builds its own entry from its own classes in its own package instead of adding fields here.
"""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.common.metadata_extension import (
    register_pytree_node_for_dataclass,
)

__all__ = [
    "AicpuGeometry",
    "AicpuMetadata",
    "AscLightningIndexerMetadata",
    "AscSparseAttentionMetadata",
    "KernelFrame",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class KernelFrame:
    """One tensor's coordinate frame: the boundaries needed to address it.

    ``cu_seqlens`` are the ragged document boundaries (``int32``, ``n_documents + 1``), the form
    the kernels take as ``cu_seqlens_*``.  ``seqused`` is the per-document prefix this frame's
    consumer addresses -- how an operator is told to read a prefix of its operand -- and
    ``residual`` the trailing partial group of a compressed axis (``None`` where the axis is not
    compressed, which the kernels reject a value for).
    """

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


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
class AscLightningIndexerMetadata:
    """What the BF16 selector's two kernels are called with: forward and backward.

    ``fwd`` is what ``lightning_indexer`` grids its scores with and ``bwd`` the one
    ``sparse_lightning_indexer_kl_loss_grad`` consumes.  A layer with no compressed axis runs no
    selector, so both are ``None`` there.  The quantized selector is another kernel pair with
    metadata of its own, carried by that backend's entry rather than by these two fields.
    """

    fwd: torch.Tensor | None
    bwd: torch.Tensor | None
    # The candidate-capable selector's two records.  Empty on this stack today: its kernels score or
    # publish the pool, but the sparse search's schedule and the count it is bound by arrive with the
    # non-quantized candidate operators.  The fields exist now so that a path which already builds
    # them (the quantized backend, in its own entry) and a path that will read them name one shape.
    fwd_sparse: torch.Tensor | None = None
    sparse_length: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class AscSparseAttentionMetadata:
    """What the AscendC attention's two kernels are called with: forward and backward."""

    fwd: torch.Tensor
    bwd: torch.Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class AicpuMetadata:
    """The AI_CPU operator outputs of one geometry, grouped by the consumer of each.

    They depend only on the frames of one forward and that geometry -- never on the operands --
    so they are computed once per microbatch (per rank under context parallelism) at the eager
    boundary and read by every layer of the geometry, in the forward, the full-AC recompute and
    the backward alike.  The tensors are shared, which is sound only while the kernels keep
    taking ``metadata`` as a read-only input: it is the operator's own output, consulted for the
    grid, never an output buffer of an NPU kernel.
    """

    geometry: AicpuGeometry
    sparse_attn: AscSparseAttentionMetadata
    lightning_indexer: AscLightningIndexerMetadata


register_pytree_node_for_dataclass(KernelFrame)
register_pytree_node_for_dataclass(AicpuGeometry)
register_pytree_node_for_dataclass(AscLightningIndexerMetadata)
register_pytree_node_for_dataclass(AscSparseAttentionMetadata)
register_pytree_node_for_dataclass(AicpuMetadata)
