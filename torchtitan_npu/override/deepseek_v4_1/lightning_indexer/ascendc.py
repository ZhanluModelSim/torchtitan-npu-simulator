# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The selector's BF16 port: LightningIndexer with its SLIKG edge.

Its two metadata tensors (``build_indexer_metadata``) are sized from the frames and the geometry
the attention override states, and neither is launched from a layer.  A backend that runs another
selector (the quantized one) builds its own entry in its own package instead of extending this one.
"""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.deepseek_v4_1.indexer import InnerIndexer
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import AscIndexerMixin
from torchtitan_npu.override.deepseek_v4_1.metadata import (
    AicpuGeometry,
    AscLightningIndexerMetadata,
)

_LAYOUT = "TND"
_MASK_MODE = 3  # right-down causal: entry j is complete for query t iff j < (t+1)//ratio


def kernel_options(*, ratio: int, q: KernelFrame, cmp_k: KernelFrame):
    """The kernel options, read off this geometry's own frames.

    Both the metadata call and the main call take these, and they must agree
    option-for-option; reading one source is what makes that structural rather than a
    convention two call sites have to keep in step.  ``cmp_k`` is the entry's frame: on arch22 the
    pool this rank's reach covers, which its keys are compacted onto.
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_k=cmp_k.cu_seqlens,
        # The per-sequence lengths actually addressed -- how an operator is told to read a prefix
        # of its operand.  Under context parallelism they carry the whole query movement:
        # ``seqused_q`` is this rank's chunk and ``seqused_k`` its reach into the gathered keys, so
        # the selection is scored at this rank's chunk against the blocks it may read.  Every
        # operator this selector can pick takes the pair -- the BF16 indexer and SLIKG -- and at
        # ``cp_size = 1`` each equals the corresponding boundary difference, so passing them is a
        # no-op there rather than a second code path.
        seqused_q=q.seqused,
        seqused_k=cmp_k.seqused,
        # A3's kernels take the zero-filled empty group; A5 rejects a value at ratio 1.
        cmp_residual_k=(
            cmp_k.residual
            if cmp_k.residual is not None or _IS_A5
            else torch.zeros(cmp_k.seqused.shape, dtype=torch.int32, device=cmp_k.seqused.device)
        ),
        layout_q=_LAYOUT,
        layout_k=_LAYOUT,
        mask_mode=_MASK_MODE,
        cmp_ratio=ratio,
    )


def build_indexer_metadata(
    *,
    ratio: int,
    q: KernelFrame,
    cmp_k: KernelFrame,
    geometry: AicpuGeometry,
) -> AscLightningIndexerMetadata:
    """The selector's two AI_CPU metadata tensors for this geometry, forward and backward.

    ``max_seqlen_q``/``max_seqlen_k`` are pinned to ``None`` because both the BF16
    ``lightning_indexer_metadata`` and SLIKG's take them as optional ints; passing them makes the
    call independent of whatever default the operator ships with.
    """
    options = kernel_options(ratio=ratio, q=q, cmp_k=cmp_k)
    geometry_kwargs = dict(
        num_heads_q=geometry.index_num_heads,
        num_heads_k=1,
        head_dim=geometry.index_head_dim,
        topk=geometry.index_topk,
        max_seqlen_q=None,
        max_seqlen_k=None,
    )
    return AscLightningIndexerMetadata(
        fwd=torch.ops.cann_ops_transformer.lightning_indexer_metadata(**geometry_kwargs, **options),
        bwd=torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            **geometry_kwargs,
            **options,
        ),
    )


def entry_options(ratio: int, metadata) -> dict:
    """The selector's two kernels' options, read off this entry's frames.

    On arch22 the entry's ``cmp_k`` is the pool this rank's reach covers, which is the frame the
    wrapper compacts the key row onto, so the operand and its grid are always one pool.
    """
    return kernel_options(ratio=ratio, q=metadata.q, cmp_k=metadata.cmp_k)


class AscInnerIndexer(AscIndexerMixin, InnerIndexer):
    """The fused BF16 LI/SLIKG selector for the V4.1 indexer layers.

    The shared LI bridge returns document-local indices and their corresponding FP32
    scores in descending index order. Trailing ``-1`` entries mark unused slots.

    It is the consumer half of the teacher pair: ``sparse_attn.asc`` emits the teacher on
    ``topk_scores``' gradient and this backward is what reads it, so training requires both overrides.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(  # pyrefly: ignore [bad-override, bad-param-name-override]
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_metadata,
        *,
        candidates_BL1C: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        indices, scores = super().forward(
            idx_q_BLHiDi,
            A3KeyWorkaround.indexer_keys(idx_k_BNDi, attention_metadata, a5=_IS_A5),
            weights_BLHi,
            attention_metadata=attention_metadata,
            options=entry_options(self.compress_ratio, attention_metadata),
            denominator=self.num_global_queries,
        )
        # BF16 selection does not produce a quantized candidate pool.
        return indices, scores, None
