# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

import functools
import logging
from typing import Any

from torchtitan.distributed import full_dtensor
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

original_post_dataloading_process = Trainer.post_dataloading_process


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels):
    """Dispatch to the model's own mask/metadata construction when present.

    Models that define ``build_attention_masks`` (e.g. DeepSeek-V4) own the
    whole per-batch metadata handling — including their own context-parallel
    sharding and plan derivation — as the single overridable seam replacing
    the removed ``mask_handler`` pattern.  Other models keep the upstream
    flow (``get_attention_masks`` + the generic ``prepare_context_parallel_input``)
    exactly as-is.  The steps after the build (the token accounting and the
    spmd-input annotation) are replicated here for the custom path.
    """
    model = self.model_parts[0]
    build = getattr(model, "build_attention_masks", None)
    if build is not None:
        inputs = input_dict["input"]
        extra_kwargs: dict[str, Any] = {k: v for k, v in input_dict.items() if k != "input"}
        cp_mesh = self.parallel_dims.get_mesh("cp") if self.parallel_dims.cp_enabled else None
        inputs, labels, extra_kwargs = build(
            inputs,
            labels,
            extra_kwargs,
            cp_mesh=cp_mesh,
            load_balancer_type=self.config.parallelism.context_parallel_load_balancer,
        )
        # Accumulate after CP sharding so labels.numel() reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()
        if self.config.parallelism.spmd_backend == "full_dtensor":
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )
        elif self.config.parallelism.spmd_backend == "spmd_types":
            inputs, labels, extra_kwargs = annotate_input_spmd_types(
                self.parallel_dims,
                inputs,
                labels,
                extra_kwargs,
            )
        return inputs, labels, extra_kwargs
    return original_post_dataloading_process(self, input_dict, labels)


def apply() -> None:
    logger.info("[PATCH] Trainer.post_dataloading_process -> patched_post_dataloading_process")
    Trainer.post_dataloading_process = patched_post_dataloading_process


apply()
