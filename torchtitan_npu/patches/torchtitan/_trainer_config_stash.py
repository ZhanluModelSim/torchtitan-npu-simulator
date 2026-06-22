# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch:
#   torchtitan.trainer.Trainer.__init__ — stash Trainer.Config and ParallelDims
#   on module-level variables for the duration of trainer construction.
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
#   Additionally, `parallel_dims` is created inside `Trainer.__init__` (via
#   `self.init_distributed()`) but is NOT passed to `config.optimizer.build()`.
#   NPU optimizer patches (Muon) need it to set up distributed communication
#   groups. We stash it by monkey-patching `Trainer.init_distributed` so that
#   the optimizer build can retrieve it via `get_active_parallel_dims()`.
#
# Idempotency:
#   Importing this module installs the wrapper exactly once thanks to Python's
#   module cache.

import functools

from torchtitan.trainer import Trainer

_trainer_config = None
_active_parallel_dims = None


def get_trainer_config():
    """Return the most recently constructed Trainer.Config.

    The config is stashed at the start of ``Trainer.__init__`` and kept
    afterwards, so both init-time patches (dataloader, loss) and training-loop
    patches (context-parallel input preparation) can read it.
    """
    return _trainer_config


def get_active_parallel_dims():
    """Return the ParallelDims created during the active Trainer.__init__.

    Returns ``None`` outside `Trainer.__init__` or before `init_distributed`
    has been called.
    """
    return _active_parallel_dims


# --- Stash Trainer.Config ---
_orig_trainer_init = Trainer.__init__


@functools.wraps(_orig_trainer_init)
def _trainer_init_with_stash(self, config, *args, **kwargs):
    global _trainer_config, _active_parallel_dims
    _trainer_config = config
    _active_parallel_dims = None
    try:
        _orig_trainer_init(self, config, *args, **kwargs)
    finally:
        _active_parallel_dims = None


Trainer.__init__ = _trainer_init_with_stash

# --- Stash ParallelDims from init_distributed ---
_orig_init_distributed = Trainer.init_distributed


def _init_distributed_with_stash(self):
    global _active_parallel_dims
    parallel_dims = _orig_init_distributed(self)
    _active_parallel_dims = parallel_dims
    return parallel_dims


Trainer.init_distributed = _init_distributed_with_stash
