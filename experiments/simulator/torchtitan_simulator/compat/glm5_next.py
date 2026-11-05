# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture support for GLM-5.3-Flash.

Two bridges are needed:

1. The runner's transitional input contract expects a model-level
   ``build_attention_masks``. GLM-5.3-Flash has maskless mixers (KDA linear
   attention and NoPE DSA; the DSA indexer consumes an all-valid token mask
   the forward builds itself), so the adapter passes the loader inputs
   through unchanged.
2. The DSA indexer's boolean-mask pooling indexing is data-dependent
   (``tensor[bool_mask]`` runs ``nonzero`` under the hood), which meta
   rejects. The capture assumes all elements non-zero, exactly like the
   reference eager fallback: shapes stay consistent while values are
   meaningless under capture anyway.
"""

from __future__ import annotations

import torch


def install_glm5_next_meta_support(scope):
    """Allow data-dependent meta indexing in the DSA indexer pooling."""
    from torch.fx.experimental import _config as fx_config

    fx_config.meta_nonzero_assume_all_nonzero = True


def _build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
    """The runner's transitional seam for the maskless GLM mixers.

    The runner synthesizes text-only batches; when the model owns a vision
    tower the capture stamps exactly one image into the row (the device
    profile packs one 448x448 image): a contiguous run of image-token ids
    sized to the merger output (grid**2 / merge**2 positions) plus the raw
    pixels and grid the tower consumes. Shape-only under meta — the grid
    values are never read (config-known uniform square)."""
    if cp_mesh is not None:
        if load_balancer_type is not None:
            raise NotImplementedError("GLM-5.3-Flash simulator capture does not use a CP load balancer")
        raise NotImplementedError("GLM-5.3-Flash simulator capture does not support context parallel")
    extra = dict(extra_inputs)
    vision = self.config.vision_encoder
    if vision is not None and "pixel_values" not in extra:
        grid = vision.image_size // vision.patch_size
        image_tokens = (grid // vision.spatial_merge_size) ** 2
        row = tokens[0]
        row[:image_tokens] = self.config.image_token_id
        if labels is not None:
            labels[0, :image_tokens] = -100
        extra["pixel_values"] = torch.zeros(1, 3, vision.image_size, vision.image_size)
        extra["grid_thw"] = torch.tensor([[1, grid, grid]], dtype=torch.long)
    return tokens, labels, extra


def install_glm5_next_input_adapter(scope):
    """Bind the runner's transitional input contract to the pass-through."""
    from torchtitan_npu.models.glm5_next.model import Glm5NextModel

    scope.replace(Glm5NextModel, "build_attention_masks", _build_attention_masks)


__all__ = ["install_glm5_next_input_adapter", "install_glm5_next_meta_support"]
