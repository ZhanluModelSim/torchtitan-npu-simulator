# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 DSA attention, reusing K3's MLA projections and output gate."""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.kimi_k3.attention import KimiGatedMLA

from .dsa import KimiDSAIndexer, selected_key_attention


class KimiGatedDSA(KimiGatedMLA):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiGatedMLA.Config):
        indexer: KimiDSAIndexer.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build()

    def _compute_attention(
        self,
        hidden_states: torch.Tensor,
        query_latent: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[1] <= self.indexer.topk:
            return super()._compute_attention(hidden_states, query_latent, query, key, value)
        indices = self.indexer(hidden_states, query_latent)
        return selected_key_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), indices, self.scaling
        )
