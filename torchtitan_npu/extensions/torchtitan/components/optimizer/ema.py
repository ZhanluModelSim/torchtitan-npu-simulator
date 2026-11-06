# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-aware offload seams for the upstream EMA implementation."""

from dataclasses import dataclass

import torch
from torchtitan.components.optim.ema import EMA as _EMA


class EMA(_EMA):
    @dataclass(kw_only=True, slots=True)
    class Config(_EMA.Config):
        pass

    def _init_cpu_offload(self) -> None:
        tensors = []
        for optimizer in self.optimizers:
            tensors.extend(state["ema_params"] for state in optimizer.state.values())
        device = next((tensor.device for tensor in tensors if tensor.device.type != "cpu"), torch.device("cpu"))
        self._offload_device = device
        if device.type != "cpu":
            self._offload_stream = torch.get_device_module(device).Stream(device=device)
        for optimizer in self.optimizers:
            for state in optimizer.state.values():
                state["ema_params"] = (
                    self._pin_local(state["ema_params"])
                    if device.type != "cpu"
                    else self._local_view(state["ema_params"])
                )

    def _update_offloaded(self, params, ema_params, decay: float) -> None:
        # Host Engram tables and accelerator parameters share one upstream
        # EMA group. Only the accelerator subset needs a staging stream.
        host_pairs = []
        device_pairs = []
        for param, ema_param in zip(params, ema_params, strict=True):
            (host_pairs if param.device.type == "cpu" else device_pairs).append((param, ema_param))
        if host_pairs:
            host_params, host_ema = zip(*host_pairs, strict=True)
            torch._foreach_lerp_(list(host_ema), list(host_params), 1.0 - decay)
        if not device_pairs:
            return
        device_params, device_ema = zip(*device_pairs, strict=True)
        device_module = torch.get_device_module(self._offload_device)
        stream = self._offload_stream
        assert stream is not None
        compute_stream = device_module.current_stream(self._offload_device)
        stream.wait_stream(compute_stream)
        params_read = device_module.Event()
        with device_module.stream(stream):
            chunks = list(self._scratch_chunks(list(device_params), list(device_ema)))
            for index, (chunk_p, chunk_e, scratch) in enumerate(chunks):
                torch._foreach_copy_(scratch, chunk_e, non_blocking=True)
                torch._foreach_lerp_(scratch, chunk_p, 1.0 - decay)
                if index == len(chunks) - 1:
                    params_read.record(stream)
                torch._foreach_copy_(chunk_e, scratch, non_blocking=True)
            self._pending_event = device_module.Event()
            self._pending_event.record(stream)
        compute_stream.wait_event(params_read)
