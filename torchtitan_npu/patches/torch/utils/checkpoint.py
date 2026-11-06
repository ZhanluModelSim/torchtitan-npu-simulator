# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Upstream-Commit: https://github.com/pytorch/pytorch/commit/17782e7b075561f13080cb9c84a253c3ffe7c5d9

"""Backport the effect predicate imported by TorchTitan graph memory policy.

This supplies the upstream predicate only. It does not add selective activation
checkpointing's effect-cache implementation to an older PyTorch runtime.
"""

import torch.utils.checkpoint


def _is_cacheable_effect(op) -> bool:
    from torch._higher_order_ops.effects import has_effects

    return has_effects(op) and op.namespace != "c10d"


def apply() -> None:
    if not hasattr(torch.utils.checkpoint, "_is_cacheable_effect"):
        torch.utils.checkpoint._is_cacheable_effect = _is_cacheable_effect  # pyrefly: ignore[missing-attribute]


apply()
