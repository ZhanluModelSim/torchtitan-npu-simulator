# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4897
# Remove this module (and its activation import in
# ``torchtitan_npu/patches/torchtitan/__init__.py``) once the pinned
# ``torchtitan`` dependency dispatches attention metadata per layer natively.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport of upstream's per-layer metadata dispatch in ``MTPDecoder``.

Upstream #4897 moved the selection of a layer's attention metadata into the
decoder's forward: the microbatch's map is built once and every layer reads the
entry its own attention backend named.  ``MTPDecoder`` overrides ``forward`` --
0.3.0's DeepSeek-V3 model and this repository's DeepSeek-V3.2 both derive from
it -- so it selects per layer as well, MTP depths included: they run the same
attention geometry as the layer they were copied from.
"""

from typing import Any

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.deepseek_v3.mtp import MTPDecoder, roll_mtp_sequence

__all__ = ["apply"]


def _layer_attention_metadata(attention_masks, attention_metadata, layer):
    """The metadata one layer reads: its own map entry, or the 0.3.0 mask.

    ``attention_masks`` keeps its pinned single-object meaning until the stacks
    that still build one mask type are migrated; passing ``attention_metadata``
    selects per layer instead (upstream #4897).
    """
    if attention_metadata is None:
        return attention_masks
    return attention_metadata.get(layer.attention.attention_metadata_key)


def forward(
    self,
    tokens: Any,
    positions: Any = None,
    attention_masks: Any = None,
    *,
    attention_metadata: Any = None,
):
    """0.3.0's ``MTPDecoder.forward`` with #4897's per-layer selection."""
    if self.mtp_layers is None:
        # The patched ``Decoder.forward`` takes the map; the pinned signature does not.
        return Decoder.forward(
            self,
            tokens,
            positions,
            attention_masks,
            attention_metadata=attention_metadata,  # pyrefly: ignore [unexpected-keyword]
        )
    if self.tok_embeddings is None:
        raise ValueError("MTP decoder forward requires token embeddings.")

    # Keep this aligned with Decoder.forward(), but preserve the pre-norm
    # hidden state because MTP consumes the last decoder-layer output.
    h = self.tok_embeddings(tokens)
    for layer in self.layers.values():
        h = layer(h, _layer_attention_metadata(attention_masks, attention_metadata, layer), positions)

    prev_depth_hidden = h
    h = self.norm(h) if self.norm is not None else h

    mtp_outputs = []
    for depth, layer in enumerate(self.mtp_layers, 1):
        # NOTE: Without SP, the local main embedding output has shape
        # [batch, seq_len, hidden_dim] and could be shifted and reused.
        # Under SP, its sequence dimension is sharded, so a local shift
        # would be incorrect at shard boundaries. Reuse in that case
        # would require a cross-shard shift or redistribution.
        mtp_input_tokens, mtp_input_valid_mask = roll_mtp_sequence(
            tokens,
            shift=depth,
            positions=positions,
            fill_value=0,
            return_valid_mask=True,
        )
        mtp_input_embed = self.tok_embeddings(mtp_input_tokens)
        prev_depth_hidden = layer(
            mtp_input_embed,
            prev_depth_hidden,
            mtp_input_valid_mask,
            _layer_attention_metadata(attention_masks, attention_metadata, layer),
            positions,
        )
        mtp_outputs.append(prev_depth_hidden)

    outputs = [h, *mtp_outputs]
    if self._skip_lm_head:
        raise ValueError("skip_lm_head is not supported with MTP decoder until ChunkedLoss supports MTP outputs.")
    return [self.lm_head(item) if self.lm_head is not None else item for item in outputs]


def apply() -> None:
    # Identity, not a marker attribute: this module replaces exactly one method,
    # and ``MTPDecoder`` inherits ``Decoder``, so a marker would have to avoid
    # colliding with the one the decoder patch leaves on the base class.  A
    # backport carries no state of its own on the class it patches.
    if MTPDecoder.forward is forward:
        return
    MTPDecoder.forward = forward  # pyrefly: ignore [bad-assignment]


apply()
