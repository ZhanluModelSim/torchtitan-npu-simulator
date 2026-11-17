# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the LICENSE file.

"""Instance-level clipping adapter for TorchTitan 0.3's module-level clip call.

Reuse the installed training step, including NPU patches, with a private globals
namespace. Remove this adapter when TorchTitan provides an instance clipping hook.
"""

from collections.abc import Iterator
from types import FunctionType

import torch
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer


class _ClippingUtils:
    def __init__(self, clipper):
        self.clip_grad_norm_ = clipper

    def __getattr__(self, name):
        return getattr(dist_utils, name)


class GradientClippingTrainer(Trainer):
    """Place after TrainerEx in the MRO so its step context wraps this adapter."""

    def clip_grad_norm(self, parameters, max_norm, **kwargs):
        return dist_utils.clip_grad_norm_(parameters, max_norm, **kwargs)

    def train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]):
        parent_step = super().train_step.__func__
        # Fail explicitly if a TorchTitan upgrade changes the clipping entry point.
        if (
            parent_step.__globals__.get("dist_utils") is not dist_utils
            or "clip_grad_norm_" not in parent_step.__code__.co_names
            or "clip_grad_norm_" in parent_step.__globals__
        ):
            raise RuntimeError("TorchTitan train_step changed; update the instance gradient clipping adapter.")
        step = FunctionType(
            parent_step.__code__,
            parent_step.__globals__ | {"dist_utils": _ClippingUtils(self.clip_grad_norm)},
            name=parent_step.__name__,
            argdefs=parent_step.__defaults__,
            closure=parent_step.__closure__,
        )
        step.__kwdefaults__ = parent_step.__kwdefaults__
        return step(self, data_iterator)
