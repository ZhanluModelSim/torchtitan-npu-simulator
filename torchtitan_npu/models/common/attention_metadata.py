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
        prepared = set()
        for layer in (*self.config.layers, *(getattr(self.config, "mtp_layers", None) or ())):
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
            # One geometry can be shared by layers with different indexer backends.
            indexer = getattr(layer.attention, "indexer", None)
            selector = getattr(indexer, "inner_indexer", None)
            prepare = getattr(selector, "prepare_attention_metadata", None)
            if prepare is not None and key in metadata and (key, type(selector)) not in prepared:
                metadata[key] = prepare(metadata[key])
                prepared.add((key, type(selector)))
        return metadata
