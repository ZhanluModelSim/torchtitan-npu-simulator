# Pending upstream PR: https://gitcode.com/Ascend/pytorch/pull/46615
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Add NPU support to Inductor's profiler-based runtime benchmarker.

The patch measures NPU callable time through the PrivateUse1 profiler fallback
and installs it as the NPU path of ``TorchProfilerBenchmarker.benchmark_gpu``.
Other device types keep the original implementation.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.runtime import benchmarking
from torchtitan.tools.logging import logger

_ORIGINAL_BENCHMARK_GPU_ATTR = "_torch_npu_original_benchmark_gpu"
_CALLABLE_PROFILE_EVENT_NAME = "_CALLABLE"
_WARMED_UP_NPU_DEVICES: set[int] = set()


def _normalize_device_type(device_type: str | torch.device | None) -> str | None:
    if device_type is None:
        return None
    if isinstance(device_type, torch.device):
        return device_type.type
    return torch.device(device_type).type


def _clear_gradients(grad_to_none: list[torch.Tensor] | None) -> None:
    if grad_to_none is not None:
        for tensor in grad_to_none:
            tensor.grad = None


def _profile_callable(
    _callable: Callable[[], Any],
    rep: int,
    buffer: torch.Tensor,
    grad_to_none: list[torch.Tensor] | None,
) -> Any:
    # Keep the PrivateUse1 fallback until the NPU Kineto backend reliably
    # collects device activities and links them to CPU regions. ProfilerStubs
    # attach elapsed device time to the _CALLABLE CPU region here.
    with torch.autograd.profiler.profile(use_device="npu", use_kineto=False) as prof:
        for _ in range(rep):
            _clear_gradients(grad_to_none)
            buffer.zero_()
            with torch.profiler.record_function(_CALLABLE_PROFILE_EVENT_NAME):
                _callable()
    return prof


def _device_time_us(event: Any) -> float:
    return max(0.0, float(getattr(event, "device_time_total", 0.0)))


@benchmarking.time_and_count
@benchmarking.gpu_benchmark_lock
def _benchmark_gpu_npu(
    self: Any,
    _callable: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    estimation_iters: int = 5,
    memory_warmup_iters: int = 10,
    max_benchmark_duration: int = 25,
    return_mode: str = "mean",
    grad_to_none: list[torch.Tensor] | None = None,
    device_type: str | torch.device | None = None,
    **kwargs: Any,
) -> float:
    del warmup, device_type, kwargs
    device_interface = get_interface_for_device("npu")

    device_interface.synchronize()
    _callable()
    device_interface.synchronize()

    buffer_size_bytes = self.get_device_cache_size("npu")
    buffer = torch.empty(buffer_size_bytes // 4, dtype=torch.int, device="npu")
    buffer.zero_()

    event_pairs = self.get_event_pairs(estimation_iters, device_type="npu")
    for start_event, end_event in event_pairs:
        _clear_gradients(grad_to_none)
        buffer.zero_()
        start_event.record()
        _callable()
        end_event.record()
    device_interface.synchronize()
    estimated_ms = self.get_event_pairs_min_timing(event_pairs)
    if estimated_ms > 0:
        rep = max(min(rep, int(max_benchmark_duration / estimated_ms)), 1)

    for _ in range(memory_warmup_iters):
        buffer.zero_()

    # Starting the fallback profiler initializes its NPU activity handler. Do
    # one discarded profiling pass so that initialization is not attributed to
    # the first measured _CALLABLE region. Initialization is process-global,
    # so only pay this cost once per device.
    device_index = device_interface.current_device()
    if device_index not in _WARMED_UP_NPU_DEVICES:
        _profile_callable(_callable, 1, buffer, grad_to_none)
        _WARMED_UP_NPU_DEVICES.add(device_index)
    prof = _profile_callable(_callable, rep, buffer, grad_to_none)

    callable_events = [event for event in prof.function_events if event.name == _CALLABLE_PROFILE_EVENT_NAME]
    if len(callable_events) != rep:
        raise AssertionError(
            "TorchProfilerBenchmarker: expected "
            f"{rep} {_CALLABLE_PROFILE_EVENT_NAME} NPU profiling events, "
            f"but found {len(callable_events)}."
        )

    callable_device_time_us = sum(_device_time_us(event) for event in callable_events)
    if callable_device_time_us <= 0:
        raise AssertionError(
            "TorchProfilerBenchmarker: NPU device time is unavailable from the PrivateUse1 profiler fallback."
        )

    avg_time_ms = (callable_device_time_us / rep) / 1000.0
    del buffer

    if return_mode in ("min", "mean", "max"):
        return avg_time_ms
    raise ValueError(f"Unsupported return_mode: {return_mode}. Use 'min', 'mean', or 'max'.")


def patch_torch_profiler_benchmarker() -> None:
    benchmarker_cls = benchmarking.TorchProfilerBenchmarker
    if hasattr(benchmarker_cls, _ORIGINAL_BENCHMARK_GPU_ATTR):
        return

    original_benchmark_gpu = benchmarker_cls.benchmark_gpu
    setattr(benchmarker_cls, _ORIGINAL_BENCHMARK_GPU_ATTR, original_benchmark_gpu)

    def benchmark_gpu(
        self: Any,
        _callable: Callable[[], Any],
        warmup: int = 25,
        rep: int = 100,
        estimation_iters: int = 5,
        memory_warmup_iters: int = 10,
        max_benchmark_duration: int = 25,
        return_mode: str = "mean",
        grad_to_none: list[torch.Tensor] | None = None,
        device_type: str | torch.device | None = None,
        **kwargs: Any,
    ) -> float:
        benchmark_impl = _benchmark_gpu_npu if _normalize_device_type(device_type) == "npu" else original_benchmark_gpu
        return benchmark_impl(
            self,
            _callable,
            warmup=warmup,
            rep=rep,
            estimation_iters=estimation_iters,
            memory_warmup_iters=memory_warmup_iters,
            max_benchmark_duration=max_benchmark_duration,
            return_mode=return_mode,
            grad_to_none=grad_to_none,
            device_type=device_type,
            **kwargs,
        )

    benchmarker_cls.benchmark_gpu = benchmark_gpu


def _register_npu_device_interface() -> None:
    try:
        # pyrefly: ignore [missing-module-attribute]
        from torch_npu.utils._dynamo import _dynamo_register_interface_for_device
    except ImportError:
        # pyrefly: ignore [missing-module-attribute]
        from torch_npu._init.registry.dynamo import _dynamo_register_interface_for_device

    _dynamo_register_interface_for_device()


def apply() -> None:
    _register_npu_device_interface()
    patch_torch_profiler_benchmarker()
    logger.info("[PATCH] TorchProfilerBenchmarker.benchmark_gpu -> torchtitan_npu NPU profiler benchmarker")
