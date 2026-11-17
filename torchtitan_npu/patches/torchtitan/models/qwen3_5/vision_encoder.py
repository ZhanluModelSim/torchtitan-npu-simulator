# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport the NPU-safe Qwen3.5 vision boundaries to pinned TorchTitan.

Upstream PR: pending publication of the Qwen3.5 vision-mask fix.
Upstream base: ``b175497ea3502388bb9513391810a86ca46dbfa8``.

The mask builder and FSDP module boundary are fixed before a model Config can
replace them, so a late Config override is insufficient. Remove when the
pinned upstream revision builds masks eagerly off CUDA and returns an owning
vision-encoder output.

Still required as of 2026-09-16: upstream wires vision mask creation through
``torch.compile(create_block_mask)`` even in eager runs, and under CANN 9.0.0
(what ``/usr/local/Ascend/cann`` resolves to on the dev container, selected
first by the launcher env chain) that compiled wrapper fails to build while
the eager path works. Only mask creation bypasses compile here; model-body
compilation (aot_eager / inductor-ascendc) is unaffected.
"""

from __future__ import annotations

from functools import wraps

from torch.nn.attention.flex_attention import create_block_mask
from torchtitan.models.qwen3_5 import vision_encoder as upstream_vision

from torchtitan_npu.extensions.torchtitan.eager_flex import (
    mark_dense_sdpa_mask_mod,
)

_ORIGINAL_MASK_BUILDER = upstream_vision.compiled_create_block_mask
_ORIGINAL_VISION_FORWARD = upstream_vision.Qwen35VisionEncoder.forward


def eager_create_block_mask(mask_mod, *args, **kwargs):
    """Use the upstream block-mask implementation without Inductor workers."""
    mark_dense_sdpa_mask_mod(mask_mod)
    kwargs["_compile"] = False
    return create_block_mask(mask_mod, *args, **kwargs)


@wraps(_ORIGINAL_VISION_FORWARD)
def owning_vision_encoder_forward(self, *args, **kwargs):
    """Return an owning tensor so FSDP keeps its pre-backward hook."""
    return _ORIGINAL_VISION_FORWARD(self, *args, **kwargs).clone()


def apply() -> None:
    """Install the two narrow vision fixes, rejecting an unknown API."""
    mask_builder = upstream_vision.compiled_create_block_mask
    vision_forward = upstream_vision.Qwen35VisionEncoder.forward
    if mask_builder not in (eager_create_block_mask, _ORIGINAL_MASK_BUILDER):
        raise RuntimeError("Unsupported upstream Qwen3.5 vision mask builder")
    if vision_forward not in (
        owning_vision_encoder_forward,
        _ORIGINAL_VISION_FORWARD,
    ):
        raise RuntimeError("Unsupported upstream Qwen3.5 vision forward")

    if mask_builder is _ORIGINAL_MASK_BUILDER:
        upstream_vision.compiled_create_block_mask = eager_create_block_mask
    if vision_forward is _ORIGINAL_VISION_FORWARD:
        upstream_vision.Qwen35VisionEncoder.forward = owning_vision_encoder_forward


__all__ = ["apply", "eager_create_block_mask", "owning_vision_encoder_forward"]
