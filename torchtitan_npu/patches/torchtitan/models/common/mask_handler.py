# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

from dataclasses import dataclass

from torchtitan.config.configurable import Configurable
from torchtitan.models.common.attention import AttentionMasksType


class BaseMaskHandler(Configurable):
    """Post-process attention masks after optional CP sharding."""

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
