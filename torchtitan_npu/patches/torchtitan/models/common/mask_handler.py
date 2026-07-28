# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

from dataclasses import dataclass

import torch

from torchtitan.config.configurable import Configurable
from torchtitan.models.common.attention import AttentionMasksType


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
        masks: AttentionMasksType,
    ) -> AttentionMasksType:
        return masks


def run_mask_handler(
    handler: BaseMaskHandler,
    masks: AttentionMasksType,
    *,
    positions: torch.Tensor | None,
) -> AttentionMasksType:
    """Run the handler with its declared position-data contract."""

    if getattr(handler, "wants_positions", False):
        return handler.post_process(masks, positions=positions)
    return handler.post_process(masks)
