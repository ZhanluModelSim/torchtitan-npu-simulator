# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/ac13e536c84e7f6647b14fa9375c3c8a8a2b8578/torchtitan/trainer.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pass packing positions to Decoder attention-mask construction."""

import torchtitan.trainer as _titan_trainer
from torchtitan.distributed.context_parallel import prepare_context_parallel_input

from torchtitan_npu.patches.torchtitan.attention import _get_position_boundary_attention

_original_trainer_post_dataloading_process = _titan_trainer.Trainer.post_dataloading_process


def _patched_trainer_post_dataloading_process(self, input_dict, labels):
    positions = input_dict.get("positions")
    # Add a position-boundary path for block-causal SDPA/Varlen attention for
    # two reasons:
    # 1. Pass positions explicitly to get_attention_masks; upstream only puts
    #    them in extra_kwargs and does not pass them to mask construction.
    # 2. Build attention_masks before prepare_context_parallel_input; CP shards
    #    the mask supplied in extra_kwargs but does not create one.
    if positions is None or _get_position_boundary_attention(getattr(self, "model_config", None)) is None:
        return _original_trainer_post_dataloading_process(self, input_dict, labels)

    # Build the mask from global dataloader positions first. CP only shards the
    # values in extra_kwargs; it does not call get_attention_masks itself.
    inputs = input_dict["input"]
    extra_inputs = {key: value for key, value in input_dict.items() if key not in ("input", "positions")}
    assert self.tokenizer is not None, "tokenizer is required for block-causal attention"
    extra_kwargs = {
        "positions": positions,
        "attention_masks": self.model_parts[0].get_attention_masks(
            input_batch=inputs,
            tokenizer=self.tokenizer,
            extra_inputs=extra_inputs,
            positions=positions,
        ),
    }

    # This path bypasses the original method, so preserve its CP step and
    # transform inputs, labels, positions, and the mask together.
    if self.parallel_dims.cp_enabled:
        inputs, labels, extra_kwargs = prepare_context_parallel_input(
            inputs,
            labels,
            extra_kwargs,
            self.parallel_dims.get_mesh("cp"),
            self.device,
            self.config.parallelism.context_parallel_load_balancer,
        )

    return inputs, labels, extra_inputs, extra_kwargs


_titan_trainer.Trainer.post_dataloading_process = _patched_trainer_post_dataloading_process
