# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AdamW with CPU canonical tensors and NPU functional updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from torch import Tensor

from torchtitan_npu.extensions.cpu_offload.runtime import (
    GradientClipChannel,
    local_tensor,
    require_cpu_tensor,
    stage_gradient,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging


@dataclass(slots=True)
class _AdamWWork:
    parameter: Tensor
    gradient: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor
    h2d: tuple[Any, ...] = ()


@dataclass(slots=True)
class _AdamWEntry:
    parameter: Tensor
    group: dict[str, Any]
    state: dict[str, Tensor]
    canonical: tuple[Tensor, Tensor, Tensor, Tensor]


class CpuOffloadAdamW(torch.optim.AdamW):
    """Keep standard AdamW state on CPU and execute each update on NPU."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        staging: CpuStaging,
        clip_channel: GradientClipChannel | None = None,
        offload_states: bool = True,
    ) -> None:
        unsupported = {
            "foreach": bool(foreach),
            "fused": bool(fused),
            "amsgrad": amsgrad,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        if option := next((name for name, enabled in unsupported.items() if enabled), None):
            raise ValueError(f"CPU-offloaded AdamW does not support {option}")

        self._staging = staging
        self._clip_channel = clip_channel
        self._offload_states = offload_states
        self._compute_device = staging.device  # pyrefly: ignore [read-only]
        self._work_buffers: dict[tuple[int, int, torch.dtype], Tensor] = {}
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            maximize=maximize,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        for group in self.param_groups:
            if option := next((name for name in unsupported if group.get(name)), None):
                raise ValueError(f"CPU-offloaded AdamW does not support {option}")

    def materialize_state(self) -> None:
        """Create the standard AdamW state directly on canonical CPU storage.

        Materialization is eager by design: the CPU-offload pipeline stages
        state to the NPU from the first step on, and checkpointing material-
        izes optimizer state the same way (``init_optim_state``), so the
        canonical storage must exist before the first ``step()`` instead of
        being created lazily on the NPU.
        """
        for group in self.param_groups:
            for parameter in group["params"]:
                require_cpu_tensor(parameter, what="AdamW parameters")
                state = self.state[parameter]
                if state:
                    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise RuntimeError("unexpected AdamW state keys")
                    if not self._offload_states:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if state[key].device.type != self._compute_device.type:
                                state[key] = state[key].to(self._compute_device)
                    continue
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = self._momentum_like(parameter)
                state["exp_avg_sq"] = self._momentum_like(parameter)

    def _momentum_like(self, parameter: Tensor) -> Tensor:
        """Allocate a moment mirroring the parameter; NPU-resident when the
        optimizer state does not join the CPU-offload plane."""
        moment = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        if not self._offload_states:
            moment = moment.to(self._compute_device)
        return moment

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.materialize_state()
        self._reserve_work_buffers()
        compute_stream = torch_npu.npu.current_stream(self._compute_device)
        transfer_stream = self._staging.stream
        entries: list[_AdamWEntry] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                local_parameter = local_tensor(parameter)
                local_gradient = local_tensor(parameter.grad)
                state = self.state[parameter]
                local_exp_avg = local_tensor(state["exp_avg"])
                local_exp_avg_sq = local_tensor(state["exp_avg_sq"])
                canonical = (
                    local_parameter,
                    local_gradient,
                    local_exp_avg,
                    local_exp_avg_sq,
                )
                if local_parameter.device.type != "cpu" or local_gradient.device.type != "cpu":
                    raise ValueError("CPU-offloaded AdamW requires CPU parameter and gradient storage")
                expected_state_device = "cpu" if self._offload_states else self._compute_device.type
                if (
                    local_exp_avg.device.type != expected_state_device
                    or local_exp_avg_sq.device.type != expected_state_device
                ):
                    raise ValueError(
                        "CPU-offloaded AdamW optimizer state must be "
                        f"{expected_state_device}-resident, got exp_avg on "
                        f"{local_exp_avg.device} and exp_avg_sq on {local_exp_avg_sq.device}"
                    )
                if any(tensor.is_complex() for tensor in canonical):
                    raise ValueError("CPU-offloaded AdamW does not support complex tensors")

                entries.append(_AdamWEntry(parameter, group, state, canonical))

        pending_work: _AdamWWork | None = None
        for index, entry in enumerate(entries):
            work = self._enqueue_work(index % 2, entry) if pending_work is None else pending_work
            next_work = self._enqueue_work((index + 1) % 2, entries[index + 1]) if index + 1 < len(entries) else None

            for handle in work.h2d:
                handle.wait_on(compute_stream)

            with torch_npu.npu.stream(compute_stream):
                # Standard AdamW update with public ops only, mirroring
                # torch.optim._functional._single_tensor_adamw for the
                # supported option subset (no foreach/fused/amsgrad/
                # capturable/differentiable; complex tensors are rejected
                # earlier; maximize is honored).
                gradient = -work.gradient if entry.group["maximize"] else work.gradient
                lr = entry.group["lr"]
                beta1, beta2 = entry.group["betas"]
                weight_decay = entry.group["weight_decay"]
                eps = entry.group["eps"]

                entry.state["step"] += 1
                step = int(entry.state["step"].item())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if weight_decay != 0:
                    work.parameter.mul_(1 - lr * weight_decay)
                work.exp_avg.lerp_(gradient, 1 - beta1)
                work.exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denom = (work.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                work.parameter.addcdiv_(work.exp_avg, denom, value=-step_size)
            if work.gradient.device.type == "npu":
                work.gradient.record_stream(compute_stream)
            self._staging.submit_d2h(
                work.parameter,
                entry.canonical[0],
                producer_stream=compute_stream,
                stream=transfer_stream,
            )
            if self._offload_states:
                self._staging.submit_d2h(
                    work.exp_avg,
                    entry.canonical[2],
                    producer_stream=compute_stream,
                    stream=transfer_stream,
                )
                self._staging.submit_d2h(
                    work.exp_avg_sq,
                    entry.canonical[3],
                    producer_stream=compute_stream,
                    stream=transfer_stream,
                )
            torch.autograd.graph.increment_version(entry.parameter)
            pending_work = next_work
        self._staging.wait()
        return loss

    def _ensure_work_buffer(self, slot: int, kind: int, numel: int, dtype: torch.dtype) -> Tensor:
        """Return the (slot, kind, dtype) work buffer, growing it if needed.

        Growth waits for in-flight staging transfers first so that the
        pre-warm below never pays a mid-pipeline stall.
        """
        key = (slot, kind, dtype)
        buffer = self._work_buffers.get(key)
        if buffer is None or buffer.numel() < numel:
            if buffer is not None:
                self._staging.wait()
            buffer = torch.empty(
                numel,
                dtype=dtype,
                device=self._compute_device,
            )
            self._work_buffers[key] = buffer
        return buffer

    def _reserve_work_buffers(self) -> None:
        """Pre-warm both slots for every tensor class before the pipelined loop."""
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter, {})
                parameter_local = local_tensor(parameter)
                tensors = [
                    (0, parameter_local),
                    (1, parameter_local),  # gradients share the parameter spec
                ]
                if self._offload_states:
                    tensors += [
                        (2, local_tensor(state["exp_avg"])),
                        (3, local_tensor(state["exp_avg_sq"])),
                    ]
                for kind, local in tensors:
                    for slot in range(2):
                        self._ensure_work_buffer(slot, kind, local.numel(), local.dtype)

    def _work_buffer(self, slot: int, kind: int, tensor: Tensor) -> Tensor:
        buffer = self._ensure_work_buffer(slot, kind, tensor.numel(), tensor.dtype)
        return buffer[: tensor.numel()].view(tensor.shape)

    def _enqueue_work(self, slot: int, entry: _AdamWEntry) -> _AdamWWork:
        parameter, gradient, exp_avg, exp_avg_sq = entry.canonical

        def gradient_destination():
            return self._work_buffer(slot, 1, gradient)

        working_gradient, gradient_handles = stage_gradient(
            entry.parameter,
            gradient,
            gradient_destination,
            staging=self._staging,
            channel=self._clip_channel,
            stream=self._staging.stream,
        )
        sources = [(0, parameter)]
        if self._offload_states:
            sources += [(2, exp_avg), (3, exp_avg_sq)]
        buffers = {kind: self._work_buffer(slot, kind, source) for kind, source in sources}
        handles = tuple(
            self._staging.submit_h2d(
                source,
                buffers[kind],
                stream=self._staging.stream,
            )
            for kind, source in sources
        )
        # NPU-resident state updates in place on the compute device and never
        # round-trips through work buffers.
        return _AdamWWork(
            buffers[0],
            working_gradient,
            buffers.get(2, exp_avg),
            buffers.get(3, exp_avg_sq),
            h2d=(*gradient_handles, *handles),
        )
