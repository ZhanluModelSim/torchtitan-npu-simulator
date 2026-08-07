# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

import functools
import logging

from torchtitan.trainer import Trainer

from torchtitan_npu.patches.torchtitan.models.common.mask_handler import (
    run_mask_handler,
)

logger = logging.getLogger(__name__)

original_post_dataloading_process = Trainer.post_dataloading_process


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels):
    """Post-process attention masks after context-parallel sharding."""
    inputs, labels, extra_kwargs = original_post_dataloading_process(
        self, input_dict, labels
    )
    masks = extra_kwargs.get("attention_masks")
    handler = getattr(self.model_parts[0], "_mask_handler", None)
    if masks is not None and handler is not None:
        extra_kwargs["attention_masks"] = run_mask_handler(
            handler, masks, positions=extra_kwargs.get("positions")
        )
    return inputs, labels, extra_kwargs


def apply() -> None:
    logger.info(
        "[PATCH] Trainer.post_dataloading_process -> patched_post_dataloading_process"
    )
    Trainer.post_dataloading_process = patched_post_dataloading_process


apply()
