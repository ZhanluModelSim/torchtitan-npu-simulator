import functools
import logging

from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

original_post_dataloading_process = Trainer.post_dataloading_process


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels):
    """Objective: call ``_mask_handler.post_process()`` on masks after CP sharding.

    The upstream ``post_dataloading_process`` returns masks that may need
    model-specific post-processing (e.g. ``BlockMask`` → sparse-attn dict).
    This patch runs the model's built ``_mask_handler`` on them.
    """
    inputs, labels, extra_kwargs = original_post_dataloading_process(
        self, input_dict, labels
    )
    masks = extra_kwargs.get("attention_masks")
    handler = getattr(self.model_parts[0], "_mask_handler", None)
    if masks is not None and handler is not None:
        extra_kwargs["attention_masks"] = handler.post_process(masks)
    return inputs, labels, extra_kwargs


def apply() -> None:
    logger.info("[PATCH] Trainer.post_dataloading_process -> patched_post_dataloading_process")
    Trainer.post_dataloading_process = patched_post_dataloading_process


apply()
