# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 selector's PyPTO port.

Same contract as :mod:`.\u0061scendc` -- LI forward, SLIKG backward, the teacher carrier and
the saved selection -- but the two kernels are the PyPTO module functions, loaded lazily on
their first NPU call.  This module is a standalone implementation of that contract: it owns
its autograd bridge and its forward, so the PyPTO runtime's calling conventions stay here and
the AscendC port is not reused or subclassed.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata,
    lightning_indexer_options,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_LI_MODULE = "torchtitan_npu.ops.pypto.lightning_indexer.lightning_indexer"
_LIG_MODULE = "torchtitan_npu.ops.pypto.sparse_lightning_indexer_kl_loss_grad.sparse_lightning_indexer_kl_loss_grad"


@cache
def _load_pypto_op(module_name: str, op_name: str, device_index: int) -> Callable[..., Any]:
    os.environ.setdefault("TILE_FWK_DEVICE_ID", str(device_index))
    return getattr(importlib.import_module(module_name), op_name)


def _device_index(tensor: torch.Tensor) -> int:
    return 0 if tensor.device.index is None else tensor.device.index


def _pypto_lightning_indexer(q: torch.Tensor, *args: Any, **kwargs: Any):
    op = _load_pypto_op(_LI_MODULE, "lightning_indexer", _device_index(q))
    return op(q, *args, **kwargs)


def _pypto_sparse_lightning_indexer_kl_loss_grad(*, q: torch.Tensor, **kwargs: Any):
    op = _load_pypto_op(
        _LIG_MODULE,
        "sparse_lightning_indexer_kl_loss_grad",
        _device_index(q),
    )
    return op(q=q, **kwargs)


class _PyPTOIndexerTND(torch.autograd.Function):
    """InnerIndexer forward fused with its SLIKG backward, over the PyPTO kernel pair."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        ratio,
        attention_metadata,
    ):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("the fused selector needs AscV4Metadata")
        li_metadata = attention_metadata.aicpu_metadata.lightning_indexer.fwd
        if li_metadata is None:
            raise ValueError("the fused selector needs the entry's forward record")
        # The selector's frame is the entry's: on arch22 the pool this rank's reach covers, which
        # is what the keys below are compacted onto, and what its records were built from.
        assert attention_metadata.cmp_k is not None, "the selector needs the entry's pool"
        options = lightning_indexer_options(ratio=ratio, q=attention_metadata.q, cmp_k=attention_metadata.cmp_k)
        topk_indices, _ = _pypto_lightning_indexer(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=li_metadata,
            return_value=1,
            **options,
        )
        # Canonicalise inside the Function: the teacher arrives on ``topk_scores``' gradient,
        # positioned per slot by whichever selection SMLAG was handed, and SLIKG consumes it
        # next to these saved indices -- sorting after ``apply`` would desynchronise the two.
        # Descending keeps position order for the valid entries and puts ``-1`` last.
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices)
        ctx.ratio = ratio
        ctx.attention_metadata = attention_metadata
        ctx.mark_non_differentiable(topk_indices)
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        del grad_indices
        idx_q, idx_k, idx_w, topk_indices = ctx.saved_tensors
        metadata = ctx.attention_metadata
        slig_metadata = metadata.aicpu_metadata.lightning_indexer.bwd
        if slig_metadata is None:
            raise ValueError("the fused selector needs the entry's backward record")
        # The emitter already applied ``softmax_scale``; divide by the **row's** query-token
        # count (the CE loss's denominator), never by this rank's slab -- under CP the
        # parameter gradients are summed across the CP axis, so a local denominator would
        # scale every indexer gradient by the CP degree.
        query_rows = metadata.swa_k.cu_seqlens[-1].to(torch.float32)
        attn_softmax_l1_norm = attn_softmax_l1_norm / query_rows
        assert metadata.cmp_k is not None, "the selector needs the entry's pool"
        options = lightning_indexer_options(ratio=ctx.ratio, q=metadata.q, cmp_k=metadata.cmp_k)
        dq, dk, dw, _ = _pypto_sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slig_metadata,
            **options,
        )
        return dq.to(idx_q.dtype), dk.to(idx_k.dtype), dw.to(idx_w.dtype), None, None, None


class PyPTOInnerIndexer(InnerIndexer):
    """The PyPTO selector: the same contract, implemented over the PyPTO kernel pair.

    It emits the kernel's document-local ``topk_indices`` in ``[B, L, K]`` layout and the
    ``topk_scores`` teacher carrier, exactly as the AscendC port does.  Nothing is inherited
    from that port: each runtime keeps its own call site.
    """

    requires_teacher = True

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("PyPTOInnerIndexer requires the entry the fused attention built.")
        topk_indices, topk_scores = _PyPTOIndexerTND.apply(
            idx_q,
            # The keys enter the Function already compacted (arch22): the gather is
            # differentiable, so the teacher's gradient scatters back onto the gathered row and
            # the wrapper's ``S(1) -> R`` reduce-scatter then lands it on the compressor.
            A3KeyWorkaround.indexer_keys(idx_k, attention_metadata, a5=_IS_A5).unsqueeze(1).contiguous(),
            idx_w.float(),
            self.index_topk,
            attention_metadata.ratio,
            attention_metadata,
        )
        return topk_indices.squeeze(1), topk_scores.squeeze(1)
