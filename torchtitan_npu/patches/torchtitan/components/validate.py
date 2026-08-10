# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

import functools
import logging

from torchtitan.components.validate import Validator

logger = logging.getLogger(__name__)

original_post_dataloading_process = Validator.post_dataloading_process


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels, model_parts):
    """Post-process attention masks before each validation forward."""
    inputs, labels, extra_kwargs = original_post_dataloading_process(
        self, input_dict, labels, model_parts
    )
    masks = extra_kwargs.get("attention_masks")
    handler = getattr(model_parts[0], "_mask_handler", None)
    if masks is not None and handler is not None:
        extra_kwargs["attention_masks"] = handler.post_process(masks)
    return inputs, labels, extra_kwargs


def apply() -> None:
    logger.info(
        "[PATCH] Validator.post_dataloading_process -> patched_post_dataloading_process"
    )
    Validator.post_dataloading_process = patched_post_dataloading_process


apply()
