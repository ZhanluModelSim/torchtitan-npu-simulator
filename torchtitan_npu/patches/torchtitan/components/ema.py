# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport EMA optimizer support from TorchTitan PR #3985.

Remove this module after the TorchTitan dependency includes the PR.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.tools import utils

EMA_OPTIMIZER = "ema_optimizer"


class _EMAParamOptimizer(Optimizer):
    """Holds ``state[p]["ema_params"]`` per parameter for one model part.

    Never step()-ed; reuses ``Optimizer``'s per-param state dict plus the
    FQN-flattening DCP machinery in ``checkpoint_utils.py`` instead of a
    bespoke DTensor state-dict format. When constructed it always wraps the
    real parameters; ``enable`` controls EMA tensor allocation.
    """

    def __init__(self, named_params: list[tuple[str, nn.Parameter]], *, enable: bool) -> None:
        params = [p for _, p in named_params]
        param_names = [canonical_fqn(name) for name, _ in named_params]
        super().__init__([{"params": params, "param_names": param_names}], {})
        if enable:
            for p in params:
                self.state[p]["ema_params"] = p.detach().clone()

    def step(self, closure: Any = None) -> Any:
        """Reject direct updates; EMA is stepped by its container."""
        raise RuntimeError(
            "_EMAParamOptimizer must not be step()-ed; call EMAOptimizersContainer.step(current_step) instead."
        )


class EMAOptimizersContainer(OptimizersContainer):
    """Pseudo-optimizer maintaining an online EMA of model weights.

    Subclasses ``OptimizersContainer`` to reuse its FQN-flattened,
    resharding-safe ``state_dict()``/``load_state_dict()`` while overriding
    ``__init__``/``step()``/``zero_grad()`` -- this is never a real training
    optimizer. Never merged into ``Trainer.optimizers`` or
    ``LRSchedulersContainer``. The NPU trainer constructs it only when EMA is
    enabled; ``enable`` also preserves a no-op configuration for direct use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore [bad-override]
        enable: bool = False
        """Whether EMA tracking is active."""

        decay: float | None = None
        """Fixed decay per firing. Dynamically computed when unset."""

        half_life_fraction: float = 0.05
        """Fraction used to compute dynamic decay when ``decay`` is unset."""

        start_step: int = 0
        """First Trainer step at which EMA tracking begins."""

        step_bias: int = 0
        """Offset used when computing the number of EMA updates."""

        update_every_n_steps: int = 1
        """Run an EMA update every N optimizer steps."""

        offload_to_cpu: bool = False
        """Keep EMA weights in pinned CPU memory."""

        def __post_init__(self) -> None:
            if self.update_every_n_steps <= 0:
                raise ValueError(f"ema_weights.update_every_n_steps must be >= 1, got {self.update_every_n_steps}")
            if self.decay is None and self.half_life_fraction <= 0:
                raise ValueError(
                    f"ema_weights.half_life_fraction must be > 0 when decay is None, got {self.half_life_fraction}"
                )
            if self.decay is not None and not 0.0 <= self.decay <= 1.0:
                raise ValueError(f"ema_weights.decay must be in [0, 1] when set, got {self.decay}")

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        self.enable = config.enable
        self.decay = config.decay
        self.half_life_fraction = config.half_life_fraction
        self.start_step = config.start_step
        self.step_bias = config.step_bias
        self.update_every_n_steps = config.update_every_n_steps
        self.offload_to_cpu = config.offload_to_cpu
        self.model_parts = model_parts

        self.optimizers: list[_EMAParamOptimizer] = []
        all_params: list[nn.Parameter] = []
        for model in model_parts:
            named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            self.optimizers.append(_EMAParamOptimizer(named_params, enable=self.enable))
            all_params.extend(p for _, p in named_params)
        self._validate_params(all_params)
        self._post_init(all_params)

        self._offload_stream: Any | None = None
        self._offload_scratch: dict[int, list[torch.Tensor]] = {}
        self._pending_event: Any | None = None
        self._param_read_event: Any | None = None
        if self.enable and self.offload_to_cpu:
            self._init_cpu_offload()

    @staticmethod
    def _local_view(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if isinstance(t, DTensor) else t

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    def step(self, current_step: int) -> None:  # pyrefly: ignore [bad-override]
        if not self.enable or current_step < self.start_step:
            return
        elapsed = current_step - self.start_step
        if elapsed % self.update_every_n_steps != 0:
            return
        # num_updates counts firings, not raw steps (equal only when
        # update_every_n_steps == 1) -- still stateless, a pure function of
        # current_step. Clamped to >= 1 for the first firing.
        num_updates = max((elapsed + self.step_bias) // self.update_every_n_steps, 1)
        self._update(num_updates)

    # --- checkpointing ---

    def state_dict(self) -> dict[str, Any]:
        if not self.enable:
            return {}
        if not self.offload_to_cpu:
            return super().state_dict()
        # Materialize real DTensors for DCP, call through, then restore the
        # pinned-CPU steady state so offload savings only lapse briefly.
        # Must restore originals even if super().state_dict() raises, or EMA
        # would remain on-device and break the offload contract.
        self._maybe_wait_pending()
        originals: list[tuple[dict[str, Any], torch.Tensor]] = []
        try:
            for ema_opt in self.optimizers:
                for p, param_state in ema_opt.state.items():
                    original = param_state["ema_params"]
                    originals.append((param_state, original))
                    param_state["ema_params"] = self._materialize_dtensor(p, original)
            return super().state_dict()
        finally:
            for param_state, original in originals:
                param_state["ema_params"] = original

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.enable:
            return
        if self.offload_to_cpu:
            self._maybe_wait_pending()
        if not state_dict:
            # Checkpoint had no EMA data (disabled at save time, or predates
            # this feature) -- cold-start from the just-loaded model weights.
            for ema_opt, model in zip(self.optimizers, self.model_parts, strict=True):
                for p in (p for p in model.parameters() if p.requires_grad):
                    source = self._local_view(p.detach()) if self.offload_to_cpu else p.detach()
                    ema_opt.state[p]["ema_params"].copy_(source)
            return
        # DCP calls our state_dict() above to build its load template, so it
        # already receives real DTensors here too -- re-pin them afterward.
        super().load_state_dict(state_dict)
        if self.offload_to_cpu:
            for ema_opt in self.optimizers:
                for param_state in ema_opt.state.values():
                    param_state["ema_params"] = self._pin_local(param_state["ema_params"])

    def wait_for_param_reads(self) -> None:
        if self._param_read_event is not None:
            utils.device_module.current_stream().wait_event(self._param_read_event)
            self._param_read_event = None

    def _decay_at(self, num_updates: int) -> float:
        if self.decay is not None:
            return self.decay
        return 2.0 ** (-1.0 / (self.half_life_fraction * num_updates))

    def _update(self, num_updates: int) -> None:
        decay = self._decay_at(num_updates)
        for part_idx, (ema_opt, model) in enumerate(zip(self.optimizers, self.model_parts, strict=True)):
            params: list[torch.Tensor] = [p for p in model.parameters() if p.requires_grad]
            if not params:
                continue
            ema_params = [ema_opt.state[p]["ema_params"] for p in params]
            if self.offload_to_cpu:
                # ema_params are pinned local-shard CPU tensors; localize
                # params too so the foreach ops never mix DTensor with Tensor.
                local_params = [self._local_view(p) for p in params]
                self._update_offloaded(part_idx, local_params, ema_params, decay)
            elif torch.is_floating_point(ema_params[0]) or torch.is_complex(ema_params[0]):
                torch._foreach_lerp_(ema_params, params, 1.0 - decay)
            else:
                for e, p in zip(ema_params, params, strict=True):
                    e.copy_(e * decay + p * (1.0 - decay))

    # --- CPU offload path (async side-stream, pinned memory) ---

    def _init_cpu_offload(self) -> None:
        self._offload_stream = utils.device_module.Stream()
        for ema_opt in self.optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"] = self._pin_local(param_state["ema_params"])

    def _pin_local(self, tensor: torch.Tensor) -> torch.Tensor:
        """DTensor has no pin_memory() dispatch support (NYI:
        aten._pin_memory.default), so pin just the local shard."""
        local = self._local_view(tensor).detach().contiguous()
        return local.cpu().pin_memory()

    def _materialize_dtensor(self, p: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_pin_local``: move the local shard back onto the
        accelerator and rewrap it as a DTensor matching ``p``'s own
        sharding, only for the duration of a checkpoint save/load -- this is
        what DCP needs to (re)shard EMA state correctly across world sizes.
        ``p`` remains the live parameter, so shape and sharding metadata are
        read directly rather than cached.
        """
        # Clone the live parameter's tensor/DTensor shell so DCP sees the same
        # global shape and placements on every rank, then fill the local shard
        # from pinned EMA storage. from_local alone has produced overlapping
        # chunk plans for small FSDP shards on Ascend (manifests as absurd
        # DCP plan sizes / NPU OOM during save).
        ema_shell = p.detach().clone()
        dest = ema_shell.to_local() if isinstance(ema_shell, DTensor) else ema_shell
        if local.numel() != dest.numel():
            raise RuntimeError(
                "EMA CPU-offload local shard numel mismatch during checkpoint "
                f"materialize: ema_local={tuple(local.shape)}/{local.numel()} "
                f"vs param_local={tuple(dest.shape)}/{dest.numel()}"
            )
        if dest.numel() > 0:
            dest.copy_(local.detach().to(dest.device, non_blocking=False).reshape_as(dest))
        return ema_shell

    def _get_scratch(self, key: int, params: list[torch.Tensor]) -> list[torch.Tensor]:
        scratch = self._offload_scratch.get(key)
        if scratch is None:
            scratch = [torch.empty_like(p) for p in params]
            self._offload_scratch[key] = scratch
        return scratch

    def _maybe_wait_pending(self) -> None:
        if self._pending_event is not None:
            self._pending_event.synchronize()
            self._pending_event = None
            self._param_read_event = None

    def _update_offloaded(
        self,
        scratch_key: int,
        params: list[torch.Tensor],
        ema_params: list[torch.Tensor],
        decay: float,
    ) -> None:
        self._maybe_wait_pending()
        # FSDP2 ranks can own 0-numel local shards. Ascend rejects size-0
        # aclrtMemcpyBatchAsync, so skip empty locals (nothing to update).
        kept = [(p, e) for p, e in zip(params, ema_params, strict=True) if p.numel() > 0]
        if not kept:
            return
        params = [p for p, _ in kept]
        ema_params = [e for _, e in kept]
        scratch = self._get_scratch(scratch_key, params)
        stream = self._offload_stream
        if stream is None:
            raise RuntimeError("EMA CPU-offload stream is not initialized")
        device_module = utils.device_module
        stream.wait_stream(device_module.current_stream())
        with device_module.stream(stream):
            torch._foreach_copy_(scratch, ema_params, non_blocking=True)  # H2D
            torch._foreach_lerp_(scratch, params, 1.0 - decay)
            self._param_read_event = device_module.Event()
            self._param_read_event.record(stream)
            torch._foreach_copy_(ema_params, scratch, non_blocking=True)  # D2H
            self._pending_event = device_module.Event()
            self._pending_event.record(stream)
        # D2H completion waits lazily at the next EMA update or checkpoint save/load.
