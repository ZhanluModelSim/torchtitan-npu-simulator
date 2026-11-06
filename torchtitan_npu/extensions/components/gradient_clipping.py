# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the LICENSE file.

"""Optimization owner with a container-specific gradient clipping seam."""

import torch
from torchtitan.components.optim import Optim
from torchtitan.distributed import utils as dist_utils

from torchtitan_npu.config.configs import OptimConfig


class GradientClippingOptim(Optim):
    """Retain Optim's finite-update checks while including Host sparse gradients."""

    Config = OptimConfig

    def _update(self, loss: torch.Tensor) -> torch.Tensor:
        # Optim owns clipping in the pinned TrainingEngine API. A sparse-table
        # container adds its externally accumulated gradients to this norm.
        clipper = getattr(self.optimizers, "clip_grad_norm", dist_utils.clip_grad_norm_)
        grad_norm = clipper(
            self.parameters,
            self.config.max_norm,
            foreach=True,
            pp_mesh=self.parallelism_context.get_optional_mesh("pp"),
            ep_enabled=self.parallelism_context.ep_enabled,
        )
        loss_is_finite = torch.isfinite(loss).all().to(torch.int32)
        if not self.parallelism_context.pp_enabled or self.pp_has_last_stage:
            loss_mesh = self.parallelism_context.get_optional_mesh("loss")
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=loss_mesh.get_group(),
                )
        pp_mesh = self.parallelism_context.get_optional_mesh("pp")
        if pp_mesh is not None:
            torch.distributed.all_reduce(
                loss_is_finite,
                op=torch.distributed.ReduceOp.MIN,
                group=pp_mesh.get_group(),
            )
        torch._assert_async(
            loss_is_finite.logical_and(torch.isfinite(grad_norm).all()),
            "Loss or gradient norm is not finite. Stopping before the update.",
        )
        self.optimizers.step()
        return grad_norm
