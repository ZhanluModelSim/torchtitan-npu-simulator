# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

import functools
import logging
from typing import Any

from torchtitan.components.validate import Validator
from torchtitan.distributed import full_dtensor

logger = logging.getLogger(__name__)

original_post_dataloading_process = Validator.post_dataloading_process


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels, model_parts):
    """Dispatch to the model's own mask/metadata construction when present
    (see the trainer patch); other models keep the upstream flow."""
    build = getattr(model_parts[0], "build_attention_masks", None)
    if build is not None:
        inputs = input_dict["input"]
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in input_dict.items() if k != "input"
        }
        cp_mesh = (
            self.parallel_dims.get_mesh("cp") if self.parallel_dims.cp_enabled else None
        )
        inputs, labels, extra_kwargs = build(
            inputs,
            labels,
            extra_kwargs,
            cp_mesh=cp_mesh,
            load_balancer_type=self.parallelism.context_parallel_load_balancer,
        )
        if self.parallelism.spmd_backend == "full_dtensor":
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )
        return inputs, labels, extra_kwargs
    return original_post_dataloading_process(self, input_dict, labels, model_parts)


def apply() -> None:
    logger.info(
        "[PATCH] Validator.post_dataloading_process -> patched_post_dataloading_process"
    )
    Validator.post_dataloading_process = patched_post_dataloading_process


apply()
