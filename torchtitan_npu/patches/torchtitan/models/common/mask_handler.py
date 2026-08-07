# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.config.configurable import Configurable


class BaseMaskHandler(Configurable):
    """Post-process attention masks after optional CP sharding."""

    wants_positions: bool = False
    """Whether ``post_process`` accepts the ``positions`` keyword."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        self.config = config

    def post_process(
        self,
        masks: Any,
        *,
        positions: torch.Tensor | None = None,
    ) -> Any:
        del positions
        return masks


def run_mask_handler(
    handler: BaseMaskHandler,
    masks: Any,
    *,
    positions: torch.Tensor | None,
) -> Any:
    """Run the handler with its declared position-data contract."""

    if getattr(handler, "wants_positions", False):
        return handler.post_process(masks, positions=positions)
    return handler.post_process(masks)
