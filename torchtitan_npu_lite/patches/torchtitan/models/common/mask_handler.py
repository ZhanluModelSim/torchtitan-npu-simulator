from dataclasses import dataclass

from torchtitan.config.configurable import Configurable
from torchtitan.models.common.attention import AttentionMasksType


class BaseMaskHandler(Configurable):
    """Base class for attention mask post-processing.

    ``post_process`` is called every step after CP sharding (or without CP)
    to transform masks before they reach the model.
    """

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
