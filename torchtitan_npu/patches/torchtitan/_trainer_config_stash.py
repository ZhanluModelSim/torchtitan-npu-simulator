# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch:
#   torchtitan.trainer.Trainer.__init__ — stash the active Trainer.Config on a
#   module-level variable for the duration of trainer construction.
#
# Why:
#   Upstream removed `JobConfig`. `Trainer.Config` is the new top-level config
#   but several non-trainer Configurables (dataloader, loss) only receive
#   narrow Config slices and runtime objects through their `build()` calls,
#   so npu patches that need MTP-related fields (`training.num_mtp_modules`,
#   `training.mtp_loss_weight`, `model_spec.name`) have no other way to read
#   them. This shared stash is consumed by the hf_datasets and loss patches.
#
# Idempotency:
#   Importing this module installs the wrapper exactly once thanks to Python's
#   module cache. Both hf_datasets and loss patches import from here.

import functools

from torchtitan.trainer import Trainer

_active_trainer_config = None


def get_active_trainer_config():
    """Return the Trainer.Config currently being used to build a Trainer.

    Returns ``None`` outside `Trainer.__init__`.
    """
    return _active_trainer_config


_orig_trainer_init = Trainer.__init__


@functools.wraps(_orig_trainer_init)
def _trainer_init_with_stash(self, config, *args, **kwargs):
    global _active_trainer_config
    _active_trainer_config = config
    try:
        _orig_trainer_init(self, config, *args, **kwargs)
    finally:
        _active_trainer_config = None


Trainer.__init__ = _trainer_init_with_stash
