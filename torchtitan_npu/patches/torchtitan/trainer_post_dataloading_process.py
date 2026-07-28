# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Patch ``Trainer.post_dataloading_process`` for dense SDPA masks."""

import torchtitan.trainer as _titan_trainer

from torchtitan_npu.patches.torchtitan.attention import _is_block_causal_sdpa_config

_original_trainer_post_dataloading_process = _titan_trainer.Trainer.post_dataloading_process


def _patched_trainer_post_dataloading_process(self, input_dict, labels):
    is_block_causal_sdpa = _is_block_causal_sdpa_config(getattr(self, "model_config", None))

    inputs, labels, extra_inputs, extra_kwargs = _original_trainer_post_dataloading_process(
        self,
        input_dict,
        labels,
    )

    if is_block_causal_sdpa and "attention_masks" not in extra_kwargs:
        if self.tokenizer is None:
            raise AssertionError("tokenizer is required for block-causal SDPA")
        model = self.model_parts[0]
        extra_kwargs["attention_masks"] = model.get_attention_masks(
            input_batch=inputs,
            tokenizer=self.tokenizer,
            extra_inputs=extra_inputs,
        )

    return inputs, labels, extra_inputs, extra_kwargs


_titan_trainer.Trainer.post_dataloading_process = _patched_trainer_post_dataloading_process
