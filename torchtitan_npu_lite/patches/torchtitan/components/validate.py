import functools
import logging

from torchtitan.components.validate import Validator

logger = logging.getLogger(__name__)

original_validate = Validator.validate


@functools.wraps(original_validate)
def patched_validate(self, model_parts, step):
    """Objective: call ``_mask_handler.post_process()`` on masks at validation.

    The upstream validation path does not run mask post-processing.
    This patch applies the same ``_mask_handler`` transform that the
    training path performs.
    """
    result = original_validate(self, model_parts, step)
    extra_kwargs = getattr(self, "extra_kwargs", {})
    masks = extra_kwargs.get("attention_masks")
    handler = getattr(model_parts[0], "_mask_handler", None)
    if masks is not None and handler is not None:
        extra_kwargs["attention_masks"] = handler.post_process(masks)
    return result


def apply() -> None:
    logger.info("[PATCH] Validator.validate -> patched_validate")
    Validator.validate = patched_validate


apply()
