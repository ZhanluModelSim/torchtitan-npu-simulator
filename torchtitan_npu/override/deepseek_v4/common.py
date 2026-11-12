# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Fused native V4 children outside the compressor-owned subtrees.

Use these entries with ``compressor.asc``. That override composes its own norm
and RoPE children; claiming those children separately conflicts with their
parent under TorchTitan's override rules.
"""

from torchtitan.config import override
from torchtitan.models.common import ComplexRoPE, RMSNorm


@override(
    target=RMSNorm.Config,
    fqns=[
        "norm",
        "model.norm",
        "*.attention_norm",
        "*.ffn_norm",
        "*.q_norm",
        "*.q_head_norm",
        "*.kv_norm",
        "*.enorm",
        "*.hnorm",
        "*.mtp_norm",
    ],
    description="Fused V4 RMSNorm outside compressor subtrees",
)
def rms_norm(cfg: RMSNorm.Config):
    from torchtitan_npu.override.common.rms_norm import asc

    return asc(cfg)


@override(
    target=ComplexRoPE.Config,
    exact=True,
    fqns=["*.attention.rope", "*.attention.indexer.rope"],
    description="Fused V4 RoPE outside compressor subtrees",
)
def rope(cfg: ComplexRoPE.Config):
    from torchtitan_npu.override.common.rope import asc_complex

    return asc_complex(cfg)
