# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

import functools
import logging
import os

import torch
from torchtitan.models.common.decoder import Decoder
from torchtitan.trainer import Trainer

from torchtitan_npu.patches.torchtitan.models.common.aux_loss import LoggedAuxLoss

logger = logging.getLogger(__name__)

original_update_from_config = Decoder.Config.update_from_config


def _resolve_global_batch_size(config) -> int:
    """Resolve the global batch size used by auxiliary-loss normalization."""
    training = config.training
    if training.global_batch_size >= 0:
        return training.global_batch_size

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    parallelism = config.parallelism
    model_parallel_degree = (
        parallelism.context_parallel_degree
        * parallelism.tensor_parallel_degree
        * parallelism.pipeline_parallel_degree
    )
    if world_size % model_parallel_degree != 0:
        raise ValueError(
            "WORLD_SIZE must be divisible by CP * TP * PP to derive the "
            "aux-loss global batch size."
        )
    batch_degree = world_size // model_parallel_degree
    training.global_batch_size = training.local_batch_size * batch_degree
    return training.global_batch_size


@functools.wraps(original_update_from_config)
def patched_update_from_config(self, *, config, **kwargs):
    """Copy the trainer's global batch size into auxiliary-loss configs."""
    original_update_from_config(self, config=config, **kwargs)

    if isinstance(config, Trainer.Config):
        global_batch_size = _resolve_global_batch_size(config)
        for _, aux_loss_cfg, _, _ in self.traverse(LoggedAuxLoss.Config):
            aux_loss_cfg.global_batch_size = global_batch_size


def apply() -> None:
    logger.info("[PATCH] Decoder.Config.update_from_config -> patched_update_from_config")
    Decoder.Config.update_from_config = patched_update_from_config


apply()
