# Pending upstream PR: https://github.com/meta-pytorch/torchft/pull/344
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport the stream and process-group registration subset of TorchFT PR 344.

Remove this patch only after the pinned TorchFT version provides these helpers,
consumer bindings and backend registration without a local override.
"""

from __future__ import annotations

__all__ = [
    "apply",
    "get_stream_context",
    "record_event",
    "synchronize",
]

import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torchft.process_group import ProcessGroup as TorchFTProcessGroup

if TYPE_CHECKING:
    from torch.distributed import PrefixStore, ProcessGroup

_INSTALLED = False


def _current_accelerator_type() -> str | None:
    if not torch.accelerator.is_available():
        return None
    accelerator = torch.accelerator.current_accelerator()
    return None if accelerator is None else accelerator.type


def get_stream_context(stream: torch.Stream | None) -> Any:
    if stream is None or (accelerator := _current_accelerator_type()) is None:
        return nullcontext()
    accelerator_module = getattr(torch, accelerator, None)
    if accelerator_module is None or not hasattr(accelerator_module, "stream"):
        return nullcontext()
    return accelerator_module.stream(stream)


def record_event() -> None:
    accelerator = _current_accelerator_type()
    if accelerator is None:
        return
    accelerator_module = getattr(torch, accelerator)
    event = accelerator_module.Event(interprocess=True) if accelerator == "cuda" else accelerator_module.Event()
    accelerator_module.current_stream().record_event(event)


def synchronize() -> None:
    if (accelerator := _current_accelerator_type()) is not None:
        getattr(torch, accelerator).current_stream().synchronize()


def _register_process_group(self: TorchFTProcessGroup, name: str) -> str:
    group_name = f"{self.getBackendName()}:{name}"

    def create_pg(
        prefix_store: PrefixStore,
        rank: int,
        world_size: int,
        timeout: float,
    ) -> ProcessGroup:
        del prefix_store, rank, world_size, timeout
        return self

    devices = ["cpu"]
    if (accelerator := _current_accelerator_type()) is not None and accelerator not in devices:
        devices.append(accelerator)
    dist.Backend.register_backend(group_name, create_pg, devices=devices)
    return group_name


def apply() -> None:
    """Install the stream helpers and process-group registration on this pin."""
    global _INSTALLED
    if _INSTALLED:
        return

    import torchft.utils as utils

    utils.get_stream_context = get_stream_context
    utils.synchronize = synchronize
    utils.record_event = record_event

    # TorchFT imports these helpers by value. Rebind consumers that were loaded
    # before this patch; modules imported later will read the patched utils.
    consumer_bindings = {
        "torchft.checkpointing.http_transport": ("get_stream_context",),
        "torchft.collectives": ("get_stream_context",),
        "torchft.futures": ("get_stream_context",),
        "torchft.manager": ("get_stream_context", "synchronize"),
        "torchft.process_group": ("get_stream_context", "record_event", "synchronize"),
    }
    replacements = {
        "get_stream_context": get_stream_context,
        "record_event": record_event,
        "synchronize": synchronize,
    }
    for module_name, names in consumer_bindings.items():
        if module := sys.modules.get(module_name):
            for name in names:
                setattr(module, name, replacements[name])

    type.__setattr__(TorchFTProcessGroup, "_register", _register_process_group)
    _INSTALLED = True


apply()
