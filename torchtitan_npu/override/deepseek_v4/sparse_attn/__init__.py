# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit SMLA replacements for TorchTitan's V4 inner-attention configs.

``asc`` preserves native outer attention and indexer projections. Its default
selector keeps TorchTitan's selection and auxiliary loss; explicitly selecting
``selector_backend='asc'`` uses the existing LI/SLIKG objective. ``asc_legacy``
retains the older NPU geometry backend's core contract.
"""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override
from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    DSV4InnerAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention

if TYPE_CHECKING:
    from .ascendc import AscCompressedSparseInnerAttention
    from .native import AscNativeInnerAttentionConfig


@override(
    target=DSV4InnerAttention.Config,
    description="Use SMLA inside native TorchTitan DeepSeek-V4 attention (TP=CP=1)",
)
def asc(
    cfg: DSV4InnerAttention.Config,
    *,
    num_heads: int = 64,
    head_dim: int = 512,
    index_num_heads: int = 64,
    index_head_dim: int = 128,
    selector_backend: str = "native",
) -> "AscNativeInnerAttentionConfig":
    """Replace SWA/HCA/CSA children while preserving native model/state layouts.

    Kernel record dimensions are explicit override arguments. The ``asc``
    selector is an opt-in training objective and requires ``aux_loss=None``.
    No CANN extension is loaded during config selection.
    """
    from .native import AscCompressedSparseAttention, AscHeavilyCompressedAttention, AscSlidingWindowAttention

    targets: dict[type[DSV4InnerAttention.Config], type[AscNativeInnerAttentionConfig]] = {
        SlidingWindowAttention.Config: AscSlidingWindowAttention.Config,
        HeavilyCompressedAttention.Config: AscHeavilyCompressedAttention.Config,
        CompressedSparseAttention.Config: AscCompressedSparseAttention.Config,
    }
    if type(cfg) not in targets:
        raise ValueError(f"sparse_attn.asc does not support {type(cfg).__qualname__}.")
    if selector_backend not in ("native", "asc"):
        raise ValueError(f"Unknown V4 selector backend: {selector_backend!r}.")
    if selector_backend == "asc" and cfg.compress_ratio == 4 and cfg.aux_loss is not None:
        raise ValueError("selector_backend='asc' has its own indexer objective; set aux_loss=None.")
    return derive(
        cfg,
        targets[type(cfg)],
        num_heads=num_heads,
        head_dim=head_dim,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
        selector_backend=selector_backend,
    )


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description="Use cann_ops_transformer TND SparseFlashMLA with fused SMLAG backward",
)
def asc_legacy(
    cfg: CompressedSparseInnerAttention.Config,
    *,
    num_heads: int = 64,
    head_dim: int = 512,
    index_num_heads: int = 64,
    index_head_dim: int = 128,
) -> "AscCompressedSparseInnerAttention.Config":
    """The fused core, with the record geometry stated on the entry that selects it.

    The four widths default to the released V4-Flash shape, so the shipped Flash recipes need no
    arguments; a recipe that runs other widths states them here
    (``asc={"num_heads":16,"index_num_heads":8}``).  The reference core carries none of them:
    only a fused port's records are sized by them, and the port config has no default for them,
    so every record is sized from a value some entry stated.  The window and the selection width
    are not stated here -- they are the layer's own config fields (``window_size`` /
    ``index_topk``), which the model states once.

    The teacher this core emits on ``topk_scores``' gradient is the raw SMLAG marginal scaled by
    ``softmax_scale``; the selector's own entry (``lightning_indexer.asc``) consumes that edge
    with SLIKG.  Select both entries together when training the indexer.
    """
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .ascendc import AscCompressedSparseInnerAttention

    # ``sharding_config`` is copied from the source by ``derive``; the
    # sharding pass already ran before the overrides (update_from_config),
    # so the derived config keeps the sharding-pass values as-is.
    result = derive(
        cfg,
        AscCompressedSparseInnerAttention.Config,
        num_heads=num_heads,
        head_dim=head_dim,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )
    return result
