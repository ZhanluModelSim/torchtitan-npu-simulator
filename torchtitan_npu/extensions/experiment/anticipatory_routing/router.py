# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic expert selection capture and replay."""

import logging

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.common.moe import TokenChoiceTopKRouter

from .cache import RoutingIndexCache, resolve_store_dtype

logger = logging.getLogger(__name__)


class _ReplaySelection:
    """Replay cached IDs after the router computes its normal selection."""

    def __init__(self, select_experts, replay_buffer):
        self.select_experts = select_experts
        self.replay_buffer = replay_buffer

    def __call__(self, *args, **kwargs):
        live_ids = self.select_experts(*args, **kwargs)
        if isinstance(live_ids, DTensor):
            replay_ids = self.replay_buffer.select_experts(live_ids.to_local())
            return DTensor.from_local(replay_ids, live_ids.device_mesh, live_ids.placements, run_check=False)
        return self.replay_buffer.select_experts(live_ids)


def build_routing_cache(model_parts, *, index_store_dtype, device=None):
    """Attach the shared cache to dynamic routers; hash tables need no replay."""
    routers = [
        (f"{i}.{name}", module)
        for i, part in enumerate(model_parts)
        for name, module in part.named_modules()
        if isinstance(module, TokenChoiceTopKRouter)
    ]
    if not routers:
        raise ValueError("Anticipatory routing requires a model with MoE routers")
    routers = [(key, router) for key, router in routers if not getattr(router, "hash", False)]
    if not routers:
        logger.warning(
            "Model has no dynamic routers: fixed hash routing needs no replay; anticipatory routing is rollback-only."
        )
    cache = RoutingIndexCache(
        store_dtype=resolve_store_dtype(index_store_dtype, max((r.num_experts for _, r in routers), default=1)),
    )
    if device is None:
        device = next(
            (parameter.device for part in model_parts for parameter in part.parameters()), torch.device("cpu")
        )
    cache.create_replay_buffers([key for key, _ in routers], device)
    for key, router in routers:
        router._anticipatory_cache = cache  # pyrefly: ignore [bad-argument-type]
        router._anticipatory_key = key  # pyrefly: ignore [bad-argument-type]
        replay_buffer = cache.replay_buffers[key]
        router._select_experts = _ReplaySelection(router._select_experts, replay_buffer)
    return cache
