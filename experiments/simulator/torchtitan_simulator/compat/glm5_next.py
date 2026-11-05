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


def install_glm5_next_meta_support(scope):
    """Allow data-dependent meta indexing in the DSA indexer pooling."""
    from torch.fx.experimental import _config as fx_config

    fx_config.meta_nonzero_assume_all_nonzero = True


def _build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
    """The runner's transitional seam for the maskless GLM mixers."""
    if cp_mesh is not None:
        if load_balancer_type is not None:
            raise NotImplementedError("GLM-5.3-Flash simulator capture does not use a CP load balancer")
        raise NotImplementedError("GLM-5.3-Flash simulator capture does not support context parallel")
    return tokens, labels, dict(extra_inputs)


def install_glm5_next_input_adapter(scope):
    """Bind the runner's transitional input contract to the pass-through."""
    from torchtitan_npu.models.glm5_next.model import Glm5NextModel

    scope.replace(Glm5NextModel, "build_attention_masks", _build_attention_masks)


__all__ = ["install_glm5_next_input_adapter", "install_glm5_next_meta_support"]
