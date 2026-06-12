# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/hf_datasets/text_datasets.py

1. Registers extra datasets (enwiki-eod, alpaca) into DATASETS.
2. Wraps HuggingFaceTextDataLoader.__init__ to extend `seq_len` by
   `num_mtp_modules` for deepseek_v32 / deepseek_v4 when Multi-Token
   Prediction is enabled.

The active Trainer.Config is recovered via `_trainer_config_stash` (shared
with the loss patch). Upstream TrainingConfig has no `num_mtp_modules`
field, so the seq_len wrapper is a no-op when the npu-side Training subclass
that introduces this field is not in use.
"""

import functools

from datasets import load_dataset
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import (
    DATASETS,
    HuggingFaceTextDataLoader,
    _process_c4_text,
)
from torchtitan.tools.logging import init_logger, logger

from ._trainer_config_stash import get_trainer_config

init_logger()

new_datasets = {
    "enwiki-eod": DatasetConfig(
        path="tests/assets/enwiki",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "alpaca": DatasetConfig(
        path="tests/assets/alpaca",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
}

added_datasets = []
for name, ds_config in new_datasets.items():
    if name not in DATASETS:
        DATASETS[name] = ds_config
        added_datasets.append(name)
        logger.info(
            "[Dataset Patch] Successfully added dataset config: %s (path: %s)",
            name,
            ds_config.path,
        )
    else:
        logger.warning("[Dataset Patch] Dataset %s already exists, skip adding", name)

if added_datasets:
    logger.info(f"[Dataset Patch] Added {len(added_datasets)} datasets in total: {added_datasets}")
    logger.info(f"[Dataset Patch] All supported datasets now: {list(DATASETS.keys())}")
else:
    logger.info(f"[Dataset Patch] No new datasets to add, current supported: {list(DATASETS.keys())}")


_MTP_ALLOWED_MODELS = frozenset({"deepseek_v32", "deepseek_v4"})


def _mtp_seq_len_delta() -> int:
    """Return the MTP seq_len bump for the active Trainer.Config, or 0.

    Raises ValueError if MTP is enabled on a model outside _MTP_ALLOWED_MODELS.
    """
    trainer_config = get_trainer_config()
    if trainer_config is None:
        return 0
    num_mtp_modules = getattr(trainer_config.training, "num_mtp_modules", 0)
    if num_mtp_modules <= 0:
        return 0
    model_spec = getattr(trainer_config, "model_spec", None)
    model_name = getattr(model_spec, "name", None) if model_spec is not None else None
    if model_name not in _MTP_ALLOWED_MODELS:
        raise ValueError(
            "Multi Token Prediction Module only can be used for "
            f"{sorted(_MTP_ALLOWED_MODELS)} model now, "
            f"got model_spec.name={model_name!r}."
        )
    return num_mtp_modules


_orig_hf_text_init = HuggingFaceTextDataLoader.__init__


@functools.wraps(_orig_hf_text_init)
def _hf_text_init_with_mtp(
    self,
    config,
    *,
    dp_world_size,
    dp_rank,
    tokenizer,
    seq_len,
    local_batch_size,
    **kwargs,
):
    seq_len = seq_len + _mtp_seq_len_delta()
    _orig_hf_text_init(
        self,
        config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        **kwargs,
    )


HuggingFaceTextDataLoader.__init__ = _hf_text_init_with_mtp
