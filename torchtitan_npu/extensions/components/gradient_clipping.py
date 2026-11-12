# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the LICENSE file.

"""Connect optimizer-owned sparse gradients to the upstream clipper seam."""

from torchtitan.components.optim import GradientClipper, Optim

from torchtitan_npu.config.configs import OptimConfig


class _SparseAwareGradientClipper(GradientClipper):
    """Delegate to the optimizer container's clipper when it owns external gradients.

    The Host sparse-table container accumulates gradients outside autograd and
    exposes ``clip_grad_norm``; using it keeps those sparse gradients in the
    same global norm. Otherwise this behaves exactly like the upstream clipper.
    """

    def __init__(self, clipper, optimizers):
        self._clipper = clipper
        self._optimizers = optimizers

    def clip_grad_norm_(self, parameters, max_norm, *, parallelism_context):
        clip = getattr(self._optimizers, "clip_grad_norm", None)
        if clip is None:
            return self._clipper.clip_grad_norm_(parameters, max_norm, parallelism_context=parallelism_context)
        return clip(
            parameters,
            max_norm,
            clipper=lambda params, norm, norm_type, **kwargs: self._clipper.clip_grad_norm_(
                params, norm, parallelism_context=parallelism_context
            ),
        )


class GradientClippingOptim(Optim):
    """Preserve upstream update checks while clipping externally owned gradients."""

    Config = OptimConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.gradient_clipper = _SparseAwareGradientClipper(self.gradient_clipper, self.optimizers)
