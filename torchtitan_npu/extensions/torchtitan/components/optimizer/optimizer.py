# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host sparse-gradient support for the native mixed-optimizer container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig

from torchtitan_npu.config import OptimizerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _sparse_grad_squared_norm(table: Any) -> torch.Tensor:
    """Owner-local sum of squares of one table's pending sparse gradient.

    Rows are partitioned across the EP group, so summing this over that group
    counts every table row exactly once. The caller only passes tables that
    have a pending gradient.
    """
    pending = table.pending_sparse_grad()
    assert pending is not None
    return pending.values().detach().float().pow(2).sum()


@torch.no_grad()
def clip_grad_norm_with_host_sparse_tables(
    parameters, max_norm, norm_type=2.0, *args, tables, clipper, scale_dense_gradients=None, **kwargs
):
    """Clip dense and Host Engram sparse gradients against one global norm.

    TorchTitan's clipper only sees parameters whose ``.grad`` is set, and the
    Host Engram table deliberately keeps its gradient out of autograd until the
    optimizer step. Without this wrapper the sparse gradient is neither counted
    in the norm nor scaled by it.

    The upstream clipper both computes the dense norm and rescales the dense
    gradients, so this runs it first and then corrects: the dense gradients get
    the ratio between the true coefficient and the dense-only one, and the
    sparse gradients get the true coefficient.
    """
    # The upstream clipper consumes its argument, so a generator would be empty
    # by the time the correction below iterates it and the dense gradients would
    # silently keep the dense-only coefficient.
    parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    dense_norm = clipper(parameters, max_norm, norm_type, *args, **kwargs)
    # A nonfinite norm would make the correction below 0/0 or NaN and
    # contaminate otherwise finite gradients before the optimizer update.
    if not torch.isfinite(dense_norm):
        raise RuntimeError("Host Engram gradient clipping: nonfinite dense gradient norm; optimizer update aborted.")

    # Every replica of a shard has to take part, including one whose tokens
    # happened to touch no row it owns, so this runs before the tables with a
    # pending gradient are selected.
    # WeakSet order depends on local object addresses. Replica collectives
    # must visit the same layer on every rank, and norm summation must be stable.
    registered_tables = sorted(tables, key=lambda table: table.layer_id)
    for table in registered_tables:
        table.reduce_sparse_gradient_across_replicas()

    tables = [table for table in registered_tables if table.pending_sparse_grad() is not None]
    if not tables:
        return dense_norm
    if norm_type != 2.0:
        raise NotImplementedError(f"Host Engram gradient clipping supports the 2-norm only, got norm_type={norm_type}.")

    group_squares = {}
    for table in tables:
        lookup_mesh = table.lookup_mesh
        group = lookup_mesh.get_group() if lookup_mesh is not None else None
        if group not in group_squares:
            group_squares[group] = torch.zeros((), dtype=torch.float32)
        group_squares[group] = group_squares[group] + _sparse_grad_squared_norm(table)
    # Preserve the existing accumulation order for tables sharing one layout;
    # separately reduce tables that opt into a different row-ownership group.
    sparse_sq = torch.zeros((), dtype=torch.float32, device=dense_norm.device)
    for group, local_sq in group_squares.items():
        local_sq = local_sq.to(dense_norm.device)
        if group is not None:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=group)
        sparse_sq = sparse_sq + local_sq

    dense_norm_f32 = dense_norm.detach().float()
    total_norm = torch.sqrt(dense_norm_f32.pow(2) + sparse_sq)
    if not torch.isfinite(total_norm):
        raise RuntimeError("Host Engram gradient clipping: nonfinite combined gradient norm; optimizer update aborted.")
    dense_coef = torch.clamp(max_norm / (dense_norm_f32 + 1e-6), max=1.0)
    total_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)

    correction = float(total_coef / dense_coef)
    if correction != 1.0:
        if scale_dense_gradients is not None:
            scale_dense_gradients(parameters, correction)
        else:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
    scale = float(total_coef)
    if scale != 1.0:
        for table in tables:
            table.scale_pending_sparse_grad(scale)
    return total_norm.to(dense_norm.dtype)


def _materialize_missing_adam_state(optimizer: torch.optim.Optimizer) -> None:
    """Create zero Adam state for parameters that have not received a gradient.

    TorchTitan initializes optimizer state lazily. Once any parameter has state,
    its current helper no longer initializes parameters that were unused in the
    step. DCP then saves a partial optimizer state dict that cannot be loaded into
    a freshly initialized optimizer. Engram training can encounter this through
    DeepSeek-V4.1 components that are inactive in a selected attention tier.

    This helper runs only while serializing. Existing state is left untouched,
    and unused parameters receive the same zero-valued state that Adam/AdamW
    would create on their first update, with a step counter of zero.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SparseAdam)):
        raise TypeError(
            f"HostSparseOptimizersContainer only supports Adam, AdamW, or SparseAdam, got {type(optimizer).__name__}."
        )

    for group in optimizer.param_groups:
        state_device = None
        if group.get("capturable") or group.get("fused"):
            state_device = group["params"][0].device

        for param in group["params"]:
            state = optimizer.state[param]
            if state:
                continue
            if isinstance(optimizer, torch.optim.SparseAdam):
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                continue
            step_device = param.device if state_device is not None else torch.device("cpu")
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            if group.get("amsgrad"):
                state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)


class HostSparseOptimizersContainer(OptimizersContainer):
    """Select SparseAdam for Host tables and complete lazy state for DCP."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizerConfig):
        def materialize(self) -> None:
            sparse_groups = [group for group in self.param_groups if group.optimizer_name == "SparseAdam"]
            OptimizerConfig.materialize(self)
            if self.name == "Muon":
                self.param_groups = sparse_groups + self.param_groups

    @staticmethod
    def _resolve_optimizer_factory(name: str) -> Callable[..., torch.optim.Optimizer]:
        if name == "SparseAdam":
            return torch.optim.SparseAdam
        return OptimizersContainer._resolve_optimizer_factory(name)

    @staticmethod
    def _build_param_groups(
        model: torch.nn.Module,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        groups, patterns = OptimizersContainer._build_param_groups(model, param_group_configs, impl_kwargs)
        for group in groups.get("SparseAdam", []):
            # SparseAdam has no fused/foreach implementation. Its group is
            # selected explicitly by the recipe, not inferred from a model type.
            group.pop("fused", None)
            group.pop("foreach", None)
            group["param_names"] = [
                canonical_fqn(name) + getattr(param, "_engram_checkpoint_suffix", "")
                for name, param in zip(group["param_names"], group["params"], strict=True)
            ]
        return groups, patterns

    def _iter_host_tables(self) -> Iterator[Any]:
        for model in self.model_parts:
            for module in model.modules():
                if getattr(module, "uses_host_offload", False):
                    yield module

    def clip_grad_norm(self, parameters, max_norm, **kwargs):
        from torchtitan.distributed.utils import clip_grad_norm_

        return clip_grad_norm_with_host_sparse_tables(
            parameters,
            max_norm,
            tables=list(self._iter_host_tables()),
            clipper=clip_grad_norm_,
            scale_dense_gradients=self._scale_dense_gradients,
            **kwargs,
        )

    def _scale_dense_gradients(self, parameters, correction: float) -> None:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for table in self._iter_host_tables():
            table.clear_sparse_gradient()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with record_function("engram::sparse_grad_attach"):
            for table in self._iter_host_tables():
                table.prepare_sparse_optimizer_step()
        result = super().step(closure=closure)
        for table in self._iter_host_tables():
            pending = table.pending_sparse_grad()
            if pending is not None:
                with record_function("engram::refresh_lookup_storage"):
                    table.refresh_lookup_storage(pending.indices()[0])
            table.mark_sparse_step_complete()
        return result

    def state_dict(self) -> dict[str, Any]:
        for optimizer in self.optimizers:
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SparseAdam)):
                _materialize_missing_adam_state(optimizer)
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer in self.optimizers:
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SparseAdam)):
                _materialize_missing_adam_state(optimizer)
        super().load_state_dict(state_dict)


__all__ = ["HostSparseOptimizersContainer", "clip_grad_norm_with_host_sparse_tables"]
