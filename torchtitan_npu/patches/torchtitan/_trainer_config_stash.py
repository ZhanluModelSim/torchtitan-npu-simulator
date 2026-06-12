# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch:
#   torchtitan.trainer.Trainer.__init__ — stash the Trainer.Config on a
#   module-level variable so non-trainer patches can read it.
#
# Why:
#   Upstream removed `JobConfig`. `Trainer.Config` is the new top-level config
#   but several non-trainer Configurables (dataloader, loss) only receive
#   narrow Config slices and runtime objects through their `build()` calls,
#   so npu patches that need MTP-related fields (`training.num_mtp_modules`,
#   `training.mtp_loss_weight`, `model_spec.name`) have no other way to read
#   them. The config is kept after construction so training-loop patches
#   (e.g. context-parallel input preparation) can read it too. This shared
#   stash is consumed by the hf_datasets, loss and mtp_context_parallel patches.
#
# Idempotency:
#   Importing this module installs the wrapper exactly once thanks to Python's
#   module cache.

import functools

from torchtitan.trainer import Trainer

_trainer_config = None


def get_trainer_config():
    """Return the most recently constructed Trainer.Config.

    The config is stashed at the start of ``Trainer.__init__`` and kept
    afterwards, so both init-time patches (dataloader, loss) and training-loop
    patches (context-parallel input preparation) can read it.
    """
    return _trainer_config


_orig_trainer_init = Trainer.__init__


@functools.wraps(_orig_trainer_init)
def _trainer_init_with_stash(self, config, *args, **kwargs):
    global _trainer_config
    _trainer_config = config
    _orig_trainer_init(self, config, *args, **kwargs)


Trainer.__init__ = _trainer_init_with_stash
