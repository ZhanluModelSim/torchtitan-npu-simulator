# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config override for HashRouter expert selection capture and replay."""

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torchtitan.models.common.moe import HashRouter

from .cache import RoutingIndexCache, RoutingReplayBuffer, resolve_store_dtype

OVERRIDE_TARGET = "torchtitan_npu.extensions.experiment.anticipatory_routing.router.anticipatory_router"


class AnticipatoryHashRouter(HashRouter):
    """Keep HashRouter.forward and replace only its dynamic expert selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(HashRouter.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self._anticipatory_cache: RoutingIndexCache | None = None
        self._anticipatory_key = ""
        self._anticipatory_replay_buffer: RoutingReplayBuffer | None = None

    def _select_experts(self, scores):  # pyrefly: ignore [bad-param-name-override]
        live_ids = super()._select_experts(scores)
        replay = self._anticipatory_replay_buffer
        if replay is None:
            return live_ids
        return replay.select_experts(live_ids)


@override(target=HashRouter.Config, exact=True, description="Capture and replay dynamic expert selection")
def anticipatory_router(config: HashRouter.Config) -> HashRouter.Config:
    # Fixed token-to-expert lookup stays in HashRouter.forward.
    if config.hash:
        return config
    return derive(config, AnticipatoryHashRouter.Config)


def configure_router_override(config):
    """Enable the factory before the parent Trainer constructs the model."""
    if not config.anticipatory.enable:
        return
    targets = [entry if isinstance(entry, str) else entry[0] for entry in config.override.imports]
    if targets.count(OVERRIDE_TARGET) > 1:
        raise ValueError("Duplicate anticipatory router override")
    if OVERRIDE_TARGET not in targets:
        config.override.imports.append(OVERRIDE_TARGET)


def build_routing_cache(model_parts, *, index_store_dtype, device=None):
    """Attach the shared cache to dynamic routers; hash tables need no replay."""
    routers = [
        (f"{i}.{name}", module)
        for i, part in enumerate(model_parts)
        for name, module in part.named_modules()
        if isinstance(module, HashRouter)
    ]
    if not routers:
        raise ValueError("Anticipatory routing requires a model with MoE routers")
    routers = [(key, router) for key, router in routers if not getattr(router, "hash", False)]
    if not routers:
        logger.warning(
            "Model has no dynamic routers: fixed hash routing needs no replay; anticipatory routing is rollback-only."
        )
    for key, router in routers:
        if not isinstance(router, AnticipatoryHashRouter):
            raise ValueError(f"Router '{key}' was not replaced by the anticipatory router override")
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
        router._anticipatory_replay_buffer = cache.replay_buffers.get(key)  # pyrefly: ignore [bad-argument-type]
    return cache
