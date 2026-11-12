# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry-keyed metadata for the DeepSeek sparse attention backends."""

from collections.abc import Hashable, Mapping
from typing import Any

AttentionMetadataMap = Mapping[Hashable, Any]


class GeometryAttentionMetadataMixin:
    config: Any

    def _get_attention_metadata(self, positions, *, padding_mask=None, max_num_documents=None, max_context_length=None):
        metadata = {}
        for layer in (*self.config.layers, *getattr(self.config, "mtp_layers", ())):
            config = layer.attention.inner_attention
            key = getattr(config, "attention_metadata_key", config._owner)
            if key not in metadata:
                item = config.build_attention_metadata(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )
                if item is not None:
                    metadata[key] = item
        return metadata
