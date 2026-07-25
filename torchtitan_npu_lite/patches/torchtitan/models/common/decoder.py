import functools
import logging

from torchtitan.models.common.decoder import Decoder
from torchtitan.trainer import Trainer

from torchtitan_npu_lite.patches.torchtitan.models.common.aux_loss import LoggedAuxLoss

logger = logging.getLogger(__name__)

original_update_from_config = Decoder.Config.update_from_config


@functools.wraps(original_update_from_config)
def patched_update_from_config(self, *, config, **kwargs):
    """Objective: syncing ``global_batch_size`` obtained from
    the ``Trainer.config`` to ``LoggedAuxLoss.Config``.
    """
    original_update_from_config(self, config=config, **kwargs)

    if isinstance(config, Trainer.Config):
        global_batch_size = config.training.global_batch_size
        assert (
            global_batch_size != -1
        ), "global_batch_size must be explicitly initialized (got -1 sentinel)"
        for _, aux_loss_cfg, _, _ in self.traverse(LoggedAuxLoss.Config):
            aux_loss_cfg.global_batch_size = global_batch_size


def apply() -> None:
    logger.info("[PATCH] Decoder.Config.update_from_config -> patched_update_from_config")
    Decoder.Config.update_from_config = patched_update_from_config


apply()
