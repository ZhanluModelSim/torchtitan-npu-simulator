# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-internal auxiliary losses: gradient injection and logging.

Auxiliary objectives computed inside the model (MoE load-balance, DSA KL, ...)
cannot reach the trainer's loss function under pipeline parallelism. The
gradient injection decouples the gradient from the scalar output, and the
logging framework accumulates per-step values with PP-safe collection.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed._functional_collectives import all_reduce
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type


def _should_run_forward() -> bool:
    """True during the true forward pass; False during AC recomputation (eager).

    Under eager activation checkpointing the recomputation runs inside a
    backward graph task.  ``_current_graph_task_id`` returns ``-1`` only
    during the actual forward -- the recomputation is skipped, avoiding
    double-counting in the accumulator.

    Under ``torch.compile`` the functionalization pass already isolates
    buffer mutations, so always return True.
    """
    if torch.compiler.is_compiling():
        return True
    return torch._C._current_graph_task_id() == -1


@spmd.register_autograd_function
class _AuxLossInjection(torch.autograd.Function):
    """Identity-forward autograd that injects an aux-loss gradient on backward."""

    @staticmethod
    def forward(ctx, carrier: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def typecheck_forward(
        carrier: torch.Tensor, aux_loss: torch.Tensor
    ) -> torch.Tensor:
        return _AuxLossInjection.apply(carrier, aux_loss)

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_carrier: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (aux_loss,) = ctx.saved_tensors
        return grad_carrier, torch.ones_like(aux_loss)


class LoggedAuxLoss(Module):
    """Accumulates per-microbatch aux losses via ``inject()`` with deferred readout.

    Subclasses call ``inject()`` each microbatch to inject the gradient and
    accumulate the scalar value.  ``read()`` returns the accumulated value
    without clearing it.  ``zero_all()`` (classmethod) reads and buffers every
    instance's accumulator, then clears the local buffers -- called from the
    optimizer step pre-hook alongside the aux-loss-free load-balancing zeroing.

    Metric keys are derived from ``type(self).__name__`` of the subclass.
    """

    # Populated during model build (before PP splitting) so counts and keys
    # are identical on every rank.  PP stages without aux loss modules still
    # join collectives via pre-allocated zero accumulators in
    # collect_aux_loss_metrics.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    # Per-step aux-loss scalars, keyed by ``(reduce_mesh, class_name)``.
    # ``zero_all()`` (optimizer step pre-hook) reads per-module accumulators
    # into this dict (overwrites previous step) and zeroes the local buffers.
    # ``collect_aux_loss_metrics()`` reads this dict at log frequency.
    _step_acc: ClassVar[dict[tuple[str, str], float]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        coeff: float
        """Aux loss coefficient. Scales the gradient contribution."""
        reduce_mesh: str = "loss"
        """Mesh to average logged values over. PP is reduced separately."""
        global_batch_size: int | None = None
        """Per-token denominator, set by ``Decoder.update_from_config``."""

    @property
    def metric_name(self) -> str:
        """PascalCase to snake_case, e.g. ``AuxLoss -> aux_loss``."""
        return re.sub(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            "_",
            type(self).__name__,
        ).lower()

    def __init__(self, config: Config):
        super().__init__()
        self.coeff = config.coeff
        self.reduce_mesh = config.reduce_mesh
        self.global_batch_size = config.global_batch_size
        self.register_buffer(
            "_acc", torch.zeros((), dtype=torch.float32), persistent=False
        )
        LoggedAuxLoss._group_counts[(config.reduce_mesh, self.metric_name)] += 1

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self._acc.device
        with torch.device(buffer_device):
            self._acc = torch.zeros((), dtype=torch.float32)

    def inject(self, carrier: torch.Tensor, raw_sum: torch.Tensor) -> torch.Tensor:
        """Inject aux loss gradient into the forward graph and accumulate for logging."""
        if self.training and _should_run_forward():
            self._acc.add_(raw_sum.detach().float() / self.global_batch_size)
        return _AuxLossInjection.apply(
            carrier, raw_sum * (self.coeff / self.global_batch_size)
        )

    def read(self) -> torch.Tensor:
        """Read the accumulated value without clearing."""
        return self._acc.detach().clone()

    @classmethod
    def zero_all(cls, model_parts) -> None:
        """Read every instance's accumulator into ``_step_acc`` and clear local buffers.

        Called from the optimizer step pre-hook (every training step).
        ``collect_aux_loss_metrics`` reads ``_step_acc`` at log frequency.
        """
        for part in model_parts:
            for block in getattr(part, "layers", {}).values():
                for module in block.modules():
                    if isinstance(module, cls):
                        key = (module.reduce_mesh, module.metric_name)
                        cls._step_acc[key] = module._acc.item()
                        module._acc.zero_()


def collect_aux_loss_metrics(model_parts, parallel_dims) -> dict[str, float]:
    """Collect and reduce aux loss metrics from all model parts for logging.

    Returns ``{name}/mean`` per group; ``{}`` when no aux losses are configured.
    """
    if not LoggedAuxLoss._group_counts:
        return {}

    pp_mesh = parallel_dims.get_optional_mesh("pp")
    local_sums = {
        key: torch.tensor(
            LoggedAuxLoss._step_acc.get(key, 0.0),
            dtype=torch.float32,
            device=device_type,
        )
        for key in LoggedAuxLoss._group_counts
    }
    metrics = {}
    for key, total in sorted(local_sums.items()):
        mesh_name, tag = key
        for mesh in (parallel_dims.get_optional_mesh(mesh_name), pp_mesh):
            if mesh is not None:
                total = all_reduce(total, reduceOp="sum", group=mesh)
        metrics[f"{tag}/mean"] = float(total.item()) / LoggedAuxLoss._group_counts[key]
    return metrics


def register_aux_loss_zero_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
):
    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: LoggedAuxLoss.zero_all(model_parts)
    )
