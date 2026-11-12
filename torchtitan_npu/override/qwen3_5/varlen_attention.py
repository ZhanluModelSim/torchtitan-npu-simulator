# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=huawei-invalid-name

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention

from torchtitan_npu.override.qwen3_5.parallelize import (
    QWEN_CONTEXT_PARALLEL_UNSUPPORTED,
)


class AscVarlenInnerAttention(UlyssesVarlenInnerAttention):
    """Refuse the not-yet-migrated context-parallel varlen attention.

    Upstream 0.4.0 folds the batch dimension into the token row: the base
    ``VarlenInnerAttention.forward`` now takes ``[T, N, H]`` and a
    ``VarlenAttentionMetadata``.  The removed implementation exchanged rank-local heads
    on pre-fold ``[B, L, N, H]`` tensors and fed CANN TND attention with
    ``actual_seq_qlen`` from the old CP metadata, so running it against the
    folded caller would mis-shard the sequence. Configuration retains the real
    Ulysses CP owner and its global-metadata preparation protocol; this forward
    keeps the explicit runtime refusal until the CP NPU validation pass.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesVarlenInnerAttention.Config):
        pass

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(QWEN_CONTEXT_PARALLEL_UNSUPPORTED)


@override(
    target=UlyssesVarlenInnerAttention.Config,
    fqns=["model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use CANN TND attention with Qwen3.5/3.6 Context Parallel",
)
def asc_cp(cfg: UlyssesVarlenInnerAttention.Config) -> AscVarlenInnerAttention.Config:
    return derive(cfg, AscVarlenInnerAttention.Config)
