# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4's PyPTO-Pro LI forward and auxiliary KL-gradient backward.

Select ``lightning_indexer.pypto_pro`` together with ``sparse_attn.asc``. The latter
builds the packed metadata and emits the teacher on ``topk_scores``' gradient;
this selector owns the LI/LIG autograd bridge. The optional PyPTO-Pro runtime
is loaded only inside the real NPU callbacks in ``ops.pypto_pro.li_lig_wrapper``.

The kernels require BF16 query/key, FP32 weights, 64 indexer heads of width
128, ratio 4 and right-down causal TND masking. The joint LI/LIG contract
supports topk 512/1024, at most 8192 query tokens and 262144 compressed keys.
The runtime ABI and upstream source attribution are recorded in
the LIG implementation in ``ops/pypto_pro/sparse_lightning_indexer_kl_loss_grad/``
and ``ops/pypto_pro/LICENSE``.
"""

from dataclasses import dataclass, fields

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
from torchtitan_npu.ops.pypto_pro.li_lig_wrapper import (
    lightning_indexer_indices,
    sparse_lightning_indexer_kl_loss_grad,
)
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata,
    lightning_indexer_options,
)


@dataclass(kw_only=True, slots=True)
class _ProMetadata(AscV4Metadata):
    """The packed frames checked before the layer graph and the versions used at kernel launch."""

    frame_tensors: tuple[torch.Tensor, ...]
    frame_versions: tuple[int, ...]
    key_tokens: int

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        local = AscV4Metadata.prepare_cp_metadata(attention_metadata, permutation=permutation)
        return _validate_compact_frames(local)


if _ProMetadata not in SUPPORTED_NODES:
    register_dataclass(_ProMetadata, serialized_type_name=f"{_ProMetadata.__module__}.{_ProMetadata.__name__}")


def _frame_tensors(metadata: AscV4Metadata) -> tuple[torch.Tensor, ...]:
    key = metadata.cmp_k
    if key is None:
        raise ValueError("PyPTO-Pro requires a compressed key frame")
    return metadata.q.cu_seqlens, metadata.q.seqused, key.cu_seqlens, key.seqused, metadata.position_k


def _validate_compact_frames(metadata: AscV4Metadata) -> _ProMetadata:
    """LI and LIG read packed boundaries and cannot limit a frame to its seqused prefix."""
    if not isinstance(metadata, AscV4Metadata):
        raise TypeError("PyPTO-Pro requires metadata from sparse_attn.asc")
    geometry = metadata.aicpu_metadata.geometry
    if (geometry.index_num_heads, geometry.index_head_dim) != (64, 128):
        raise ValueError("PyPTO-Pro LI/LIG training requires 64 indexer heads of width 128")
    tensors = _frame_tensors(metadata)
    q_cu, q_used, k_cu, k_used, positions = tensors
    if (
        q_cu.ndim != 1
        or q_cu.numel() < 2
        or k_cu.shape != q_cu.shape
        or q_used.shape != (q_cu.numel() - 1,)
        or k_used.shape != q_used.shape
    ):
        raise ValueError("PyPTO-Pro compact frames require one length per document")
    q_lengths, k_lengths = torch.diff(q_cu), torch.diff(k_cu)
    compact = (
        (q_cu[0] == 0)
        & (k_cu[0] == 0)
        & (q_cu[-1] == positions.numel())
        & (q_lengths >= 0).all()
        & (k_lengths >= 0).all()
        & (q_used == q_lengths).all()
        & (k_used == k_lengths).all()
    )
    # Read the extent and check all conditions in one device synchronization.
    key_tokens = int(torch.where(compact, k_cu[-1], -1))
    if key_tokens < 0:
        raise ValueError("PyPTO-Pro requires compact query and key frames covering the packed operands")
    try:
        versions = tuple(tensor._version for tensor in tensors)
    except RuntimeError as error:
        raise ValueError(
            "PyPTO-Pro frame tensors need tracked versions; build metadata outside inference_mode"
        ) from error
    return _ProMetadata(
        **{field.name: getattr(metadata, field.name) for field in fields(AscV4Metadata)},
        frame_tensors=tensors,
        frame_versions=versions,
        key_tokens=key_tokens,
    )


def _pro_options(metadata: _ProMetadata, ratio: int) -> dict:
    if any(
        checked is not current
        for checked, current in zip(metadata.frame_tensors, _frame_tensors(metadata), strict=True)
    ):
        raise ValueError("PyPTO-Pro frame tensors were replaced; rebuild metadata before launching LI/LIG")
    key = metadata.cmp_k
    assert key is not None
    options = lightning_indexer_options(ratio=ratio, q=metadata.q, cmp_k=key)
    # The real opaque callbacks also reject in-place changes using host version counters.
    del options["seqused_q"], options["seqused_k"]
    options["guard_tensors"] = list(metadata.frame_tensors)
    options["guard_versions"] = list(metadata.frame_versions)
    return options


class _PyPTOProIndexerTND(torch.autograd.Function):
    """Keep the saved selection and the SMLAG teacher in the same slot order."""

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
        if not isinstance(attention_metadata, _ProMetadata):
            raise TypeError("PyPTO-Pro requires the compact metadata prepared by its indexer config")
        li_metadata = attention_metadata.aicpu_metadata.lightning_indexer.fwd
        if li_metadata is None:
            raise ValueError("the fused selector needs the entry's forward record")
        torch._check(idx_q.shape[0] == attention_metadata.position_k.numel())
        torch._check(idx_k.shape[0] == attention_metadata.key_tokens)
        topk_indices, topk_scores = lightning_indexer_indices(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=li_metadata,
            **_pro_options(attention_metadata, ratio),
        )
        # Sort before saving: SMLAG emits the teacher in the returned slot order.
        # Descending puts -1 padding after valid document-local positions.
        topk_indices, order = topk_indices.sort(dim=-1, descending=True)
        topk_scores = topk_scores.gather(-1, order).float().masked_fill(topk_indices < 0, -torch.inf)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices)
        ctx.ratio = ratio
        ctx.attention_metadata = attention_metadata
        ctx.mark_non_differentiable(topk_indices)
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        del grad_indices
        if attn_softmax_l1_norm is None:
            return (None,) * 6
        idx_q, idx_k, idx_w, topk_indices = ctx.saved_tensors
        metadata = ctx.attention_metadata
        slig_metadata = metadata.aicpu_metadata.lightning_indexer.bwd
        if slig_metadata is None:
            raise ValueError("the fused selector needs the entry's backward record")
        dq, dk, dw, _ = sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slig_metadata,
            **_pro_options(metadata, ctx.ratio),
        )
        return dq.to(idx_q.dtype), dk.to(idx_k.dtype), dw.to(idx_w.dtype), None, None, None


class PyPTOProInnerIndexer(InnerIndexer):
    """Standalone selector over compile-safe, lazily loaded PyPTO-Pro kernels."""

    requires_teacher = True

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        @staticmethod
        def prepare_attention_metadata(metadata: AscV4Metadata) -> _ProMetadata:
            return _validate_compact_frames(metadata)

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("PyPTOProInnerIndexer requires the entry the fused attention built.")
        topk_indices, topk_scores = _PyPTOProIndexerTND.apply(
            idx_q,
            # index_select stays outside the Function so dk scatters back to
            # the gathered whole-row keys before the CP wrapper reduce-scatters.
            A3KeyWorkaround.indexer_keys(idx_k, attention_metadata).unsqueeze(1).contiguous(),
            idx_w.float(),
            self.index_topk,
            attention_metadata.ratio,
            attention_metadata,
        )
        return topk_indices.squeeze(1), topk_scores.squeeze(1)
