# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 configurations, derived from the shared K3 geometry builders."""

from dataclasses import fields
from functools import partial

from torchtitan_npu.models.kimi_k3 import _linear_config, kimi_k3_configs
from torchtitan_npu.models.kimi_k3.attention import KimiGatedMLA

from .attention import KimiGatedDSA
from .dsa import KimiDSAIndexer, KimiIndexerLayerNorm
from .model import KimiK31Model


def _model_config(flavor: str) -> KimiK31Model.Config:
    base = kimi_k3_configs[flavor]()
    for layer in base.layers:
        attention = layer.attention
        if not isinstance(attention, KimiGatedMLA.Config):
            continue
        heads, width, topk = (2, 16, 8) if flavor == "debug" else (32, 128, 2048)
        layer.attention = KimiGatedDSA.Config(
            **{f.name: getattr(attention, f.name) for f in fields(attention) if f.init},
            indexer=KimiDSAIndexer.Config(
                wq_b=_linear_config(attention.q_lora_rank, heads * width),
                wk=_linear_config(attention.dim, width),
                weights_proj=_linear_config(attention.dim, heads),
                k_norm=KimiIndexerLayerNorm.Config(dim=width, eps=base.norm_eps),
                num_heads=heads,
                head_dim=width,
                topk=topk,
            ),
        )
    return KimiK31Model.Config(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})


kimi_k3_1_configs = {flavor: partial(_model_config, flavor) for flavor in ("debug", "16layer_reduced", "full")}
